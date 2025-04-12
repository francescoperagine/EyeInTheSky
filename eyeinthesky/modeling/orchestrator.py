from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import one_cycle
import torch
import torch.optim as optim
from eyeinthesky.modeling.trainer import MergedClassDetectionTrainer
from eyeinthesky.modeling.distillation import DistillationCallback
from eyeinthesky.modeling.scheduler import DecayingCyclicalLRSchedulerCallback
from eyeinthesky.visualization import FeatureVisualizationCallback

class DistillationOrchestrator:
    """
    Orchestrates the knowledge distillation process between teacher and student YOLO models.
    
    This class handles the setup and coordination of all components needed for distillation:
    - Initializes and connects teacher and student models
    - Sets up feature extraction hooks at specified network layers
    - Configures feature adaptation modules to match dimensions between models
    - Registers callbacks for distillation loss calculation and visualization
    - Manages learning rate scheduling (standard or cyclical)
    
    The orchestrator follows a callback-based architecture where different
    components hook into the training process at specific points (epoch start/end,
    training start, etc.) without modifying the core training loop.
    
    Most configuration parameters are passed through kwargs, which typically come from
    a YAML config file (e.g., config["distillation"] section). This allows for flexible
    configuration without changing code.
    
    Key processes:
    1. Pre-computing channel dimensions to create appropriate feature adapters
    2. Setting up distillation callbacks to capture and compare features
    3. Configuring learning rate schedulers (optional cyclical scheduler)
    4. Managing feature visualization at specified intervals
    
    Args:
        teacher_trainer (MergedClassDetectionTrainer): Trainer instance with teacher model
        student_trainer (MergedClassDetectionTrainer): Trainer instance with student model
        **kwargs: Configuration parameters typically from config["distillation"], including:
            feature_layers (list): Layer indices from which to extract features for distillation
            temperature (float): Temperature parameter for softening importance weights
            alpha (float): Weight balancing detection loss vs. distillation loss (0-1)
            use_cyclical_lr (bool): Whether to use cyclical learning rate scheduling
            max_lr (float): Maximum learning rate for cyclical scheduler
            cycle_size (int): Number of epochs per learning rate cycle
            group_scalers (dict): Per-parameter group learning rate multipliers
            feature_plot_interval (int): Interval (in epochs) for feature map visualization
    
    Example:
        ```
        teacher_trainer = MergedClassDetectionTrainer(teacher_config)
        student_trainer = MergedClassDetectionTrainer(student_config)
        
        # Load distillation settings from config
        orchestrator = DistillationOrchestrator(
            teacher_trainer=teacher_trainer,
            student_trainer=student_trainer,
            **config["distillation"]
        )
        
        # Start distillation training
        results = orchestrator.start()
        ```
    """

    def __init__(
            self, 
            teacher_trainer : MergedClassDetectionTrainer, 
            student_trainer : MergedClassDetectionTrainer, 
            **kwargs
        ):
        """Initialize with separate trainers for teacher and student"""
        self.teacher_trainer = teacher_trainer
        self.student_trainer = student_trainer
        
        self.feature_layers = kwargs.get("feature_layers", [20])
        self.temperature = kwargs.get("temperature", 2.0)
        self.alpha = kwargs.get("alpha", 0.5)
        self.use_cyclical_lr = kwargs.get("use_cyclical_lr", False)
        self.max_lr = kwargs.get("max_lr", 0.01)
        self.cycle_size = kwargs.get("cycle_size", 20)
        self.group_scalers = kwargs.get("group_scalers", {0: 1.0, 1: 1.0, 2: 1.0})
        self.feature_plot_interval = kwargs.get("feature_plot_interval", 20)
        
        # Access models for convenience
        self.teacher_model = YOLO(self.teacher_trainer.args.model)
        self.student_model = YOLO(self.student_trainer.args.model)

        self.layer_channels = self._compute_all_channel_dimensions(self.feature_layers)
        
        if not self.feature_layers:
            return
            
        # Create and register the distillation callback
        self.distill_callback = DistillationCallback(
            teacher_model=self.teacher_model,
            temperature=self.temperature,
            alpha=self.alpha,
            feature_layers=self.feature_layers,
            layer_channels=self.layer_channels
        )
        
        # Add the callback to the student trainer
        self.student_trainer.add_callback("on_train_start", self.distill_callback.init_parameters)
        self.student_trainer.add_callback("on_train_epoch_start", self.distill_callback.setup_criterion)
        self.student_trainer.add_callback("on_fit_epoch_end", self.distill_callback.log_metrics)

        # Visualization callback

        visualization_callback = FeatureVisualizationCallback(layers=self.feature_layers, interval=self.feature_plot_interval)

        self.student_trainer.add_callback("on_train_epoch_start", visualization_callback.set_hooks)
        self.student_trainer.add_callback("on_train_epoch_end", visualization_callback.plot_figures)


        if self.use_cyclical_lr:

            cyclical_lr_scheduler_callback = DecayingCyclicalLRSchedulerCallback(
                max_lr=self.max_lr,
                cycle_size=self.cycle_size,
                group_scalers=self.group_scalers
            )
            
            student_trainer.add_callback("on_pretrain_routine_end", cyclical_lr_scheduler_callback.set_scheduler)
        else:
            # Refresh default scheduler to ensure all parameter groups are included

            def refresh_scheduler_callback(trainer):
                """
                Refresh the scheduler once after all parameter groups have been added
                to ensure proper learning rate scheduling for all groups.
                """
                # Default YOLO scheduler with cosine lr - one_cycle
                trainer.lf = one_cycle(1, trainer.args.lrf, trainer.epochs)  # cosine 1->hyp['lrf']
                trainer.scheduler = optim.lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda=trainer.lf)
                
                # Restore scheduler state
                trainer.scheduler.last_epoch = trainer.start_epoch - 1
                
                LOGGER.info(f"Refreshed scheduler to include all {len(trainer.optimizer.param_groups)} parameter groups")
            student_trainer.add_callback("on_train_start", refresh_scheduler_callback)

    def _compute_all_channel_dimensions(self, feature_layers):
        """Pre-compute channel dimensions for all feature layers"""
        LOGGER.info(f"Computing channel dimensions for {len(feature_layers)} layers...")
        
        # Dictionary to store channel dimensions
        layer_channels = {}
        
        # Function to extract channels from a single model
        def get_model_channels(model, name):
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            
            # Create a very small dummy input to minimize computation
            dummy_input = torch.zeros(1, 3, self.student_trainer.args.imgsz, self.student_trainer.args.imgsz, device=device, dtype=dtype)
            
            # Store activation outputs
            activations = {}
            
            # Define a clean hook function
            def hook_func(idx):
                def hook(module, input, output):
                    activations[idx] = output
                return hook
            
            # Register hooks
            handles = []
            for idx in feature_layers:
                handles.append(model.model.model[idx].register_forward_hook(hook_func(idx)))
            
            # Do a single forward pass with no gradient computation
            with torch.no_grad():
                model.eval()  # Ensure model is in eval mode
                _ = model(dummy_input)
            
            # Get channels and clean up
            channels = {}
            for idx, activation in activations.items():
                channels[idx] = activation.shape[1]  # Channel dimension
            
            # Remove hooks
            for handle in handles:
                handle.remove()
            
            return channels
        
        # Get channels for both models
        student_channels = get_model_channels(self.student_model, "student")
        teacher_channels = get_model_channels(self.teacher_model, "teacher")
        
        # Combine and format results
        for layer_idx in feature_layers:
            s_ch = student_channels.get(layer_idx)
            t_ch = teacher_channels.get(layer_idx)
            
            if s_ch is not None and t_ch is not None:
                layer_channels[layer_idx] = {
                    'student': s_ch,
                    'teacher': t_ch
                }
                LOGGER.info(f"Layer {layer_idx}: Student channels = {s_ch}, Teacher channels = {t_ch}")
            else:
                LOGGER.warning(f"Missing channel info for layer {layer_idx}")
        
        return layer_channels

    def start(self):
        """Run distillation training by invoking student trainer once"""
        LOGGER.info(f"Distillation trainer initialized with temperature={self.temperature}, alpha={self.alpha}")
        
        return self.student_trainer.train()