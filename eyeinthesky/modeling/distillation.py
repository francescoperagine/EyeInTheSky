from ultralytics.utils import LOGGER
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from eyeinthesky.modeling.feature import FeatureAdaptation

class DistillationCallback:
    """
    Core callback that implements the knowledge distillation process.
    
    This callback manages the entire feature-based distillation mechanism by:
    
    1. Capturing intermediate features from both teacher and student models
    2. Creating and managing feature adaptation layers to align dimensions
    3. Computing cosine similarity losses between adapted features
    4. Balancing detection loss with feature distillation loss
    5. Learning optimal layer importance weights and loss component weights
    
    The callback hooks into multiple phases of training:
    - on_train_start: Initializes parameters and adds them to the optimizer
    - on_train_epoch_start: Sets up feature extraction hooks and custom loss function
    - on_fit_epoch_end: Logs metrics and visualizations to W&B
    
    Args:
        teacher_model: Pretrained YOLO model to use as knowledge source
        temperature (float): Controls softness of weight distribution (from config)
        alpha (float): Balance between detection and distillation loss (from config)
        feature_layers (list): Layer indices to extract features from (from config)
        layer_channels (dict): Pre-computed channel dimensions for each layer
    
    The callback dynamically learns two sets of weights during training:
    - layer_weights: Importance of each feature layer in the distillation loss
    - component_weights: Automatic balancing of box/class/DFL loss components
    """
    
    def __init__(self, 
            teacher_model, 
            temperature = 2.0, 
            alpha = 0.5, 
            feature_layers = [11, 14, 17, 20], 
            layer_channels = {}
        ):

        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.feature_layers = feature_layers
        self.layer_channels = layer_channels
        self.current_features = {}
        self.teacher_hooks = {}
        self.student_hooks = {}
        self.adapters = {}
        self.layer_weight_values = {layer_idx: 0.0 for layer_idx in self.feature_layers}
        self.cosine_losses = []
        self.detection_losses = []
        self.distillation_losses = []
        self.layer_weights = None
        self.component_weights = None
        self.component_weight_history = {'box': [], 'cls': [], 'dfl': []}
        self.initialized = False

    def init_parameters(self, trainer):
        """Initialize parameters and register them with optimizer once"""
        if self.initialized:
            return
            
        # Move teacher to student's device/dtype
        device = next(self.teacher_model.parameters()).device
        dtype = next(self.teacher_model.parameters()).dtype
        self.teacher_model = self.teacher_model.to(device=device, dtype=dtype)
        
        # Initialize component weights
        self.component_weights = nn.Parameter(torch.zeros(3, device=device))
        self._add_optimizer_param_group(
            trainer=trainer,
            params=[self.component_weights],
            name='loss_component_weights',
            weight_decay_multiplier=0.2
        )
            
        # Initialize layer weights
        equal_weight = 1.0 / len(self.feature_layers)
        weights_tensor = torch.tensor([equal_weight for _ in self.feature_layers], device=device)
        self.layer_weights = nn.Parameter(weights_tensor, requires_grad=True)
        self._add_optimizer_param_group(
            trainer=trainer,
            params=[self.layer_weights],
            name='layer_weight_params',
            weight_decay_multiplier=1.0
        )
        
        # Create feature adapters
        for layer_idx in self.feature_layers:
            adapter_key = f'adapter_{layer_idx}'
            
            # Get channel dimensions from pre-computed values
            if layer_idx in self.layer_channels:
                student_channels = self.layer_channels[layer_idx]['student']
                teacher_channels = self.layer_channels[layer_idx]['teacher']
                
                # Create feature adaptation layer with normalization
                self.adapters[adapter_key] = FeatureAdaptation(student_channels, teacher_channels, layer_idx=layer_idx).to(device)
                
                # Add adapter parameters to optimizer with stronger regularization
                self._add_optimizer_param_group(
                    trainer=trainer,
                    params=list(self.adapters[adapter_key].parameters()),
                    name=f'adapter_params_{layer_idx}',
                    weight_decay_multiplier=5.0
                )
            else:
                LOGGER.warning(f"Missing channel dimensions for layer {layer_idx}")
                
        self.initialized = True
        
        # Set original criterion if not already done
        if not hasattr(trainer.model, "original_criterion"):
            trainer.model.original_criterion = trainer.model.init_criterion()
    
    def setup_criterion(self, trainer):
        """Register hooks for feature extraction"""
        # Clear features dictionary
        self.current_features = {}

        def get_features(name, layer_idx):
            def hook(module, input, output):
                if name not in self.current_features:
                    self.current_features[name] = {}
                self.current_features[name][layer_idx] = output.detach()
            return hook
       
        # Register hooks for both models at each feature layer
        for layer_idx in self.feature_layers:
            self.teacher_hooks[layer_idx] = self.teacher_model.model.model[layer_idx].register_forward_hook(
                get_features('teacher', layer_idx)
            )
            self.student_hooks[layer_idx] = trainer.model.model[layer_idx].register_forward_hook(
                get_features('student', layer_idx)
            )
        
        # Define distillation criterion
        def distillation_criterion(preds, batch):

            # Get original loss
            original_loss, loss_items = trainer.model.original_criterion(preds, batch)

            # Apply uncertainty weighting (Kendall et al. 2018)
            # Converting parameters to precisions using exponential - ensures positive weights
            precision = torch.exp(-self.component_weights)

            # Weight losses - dividing by precision penalizes high uncertainty components
            weighted_loss = original_loss * precision + 0.5 * self.component_weights  # The 0.5 * log_var term is from the math derivation
            weighted_detection_loss = weighted_loss.sum()

            # Run teacher forward pass to capture features
            with torch.no_grad():
                self.teacher_model.eval()
                teacher_input = batch["img"].float()
                _ = self.teacher_model.model(teacher_input)
            
            # Calculate cosine loss for each layer and combine
            layer_losses = []
            
            for i, layer_idx in enumerate(self.feature_layers):
                if ('teacher' in self.current_features and 
                    'student' in self.current_features and
                    layer_idx in self.current_features['teacher'] and
                    layer_idx in self.current_features['student']):
                    
                    t_feat = self.current_features['teacher'][layer_idx]
                    s_feat = self.current_features['student'][layer_idx]
                    
                    # Apply adaptation to student features
                    adapter_key = f'adapter_{layer_idx}'
                    if adapter_key in self.adapters:
                        s_feat_adapted = self.adapters[adapter_key](s_feat)
                        
                        # Normalize features
                        t_feat = F.normalize(t_feat.flatten(1), p=2, dim=1)
                        s_feat_adapted = F.normalize(s_feat_adapted.flatten(1), p=2, dim=1)
                        
                        # Cosine similarity loss for this layer
                        layer_cosine_loss = 1 - F.cosine_similarity(s_feat_adapted, t_feat, dim=1).mean()
                        layer_losses.append(layer_cosine_loss)

            # Apply softmax to weights with minimum weight guarantee
            alpha = 0.2  # This is your minimum weight parameter (not the same as distillation alpha)
            uniform_weights = torch.ones_like(self.layer_weights) / len(self.layer_weights)
            normalized_weights = (1 - alpha) * F.softmax(self.layer_weights / self.temperature, dim=0) + alpha * uniform_weights
            # normalized_weights = F.softmax(self.layer_weights / self.temperature, dim=0)

            # Calculate weighted loss
            total_cosine_loss = sum(weight * loss for weight, loss in zip(normalized_weights, layer_losses))

            # Combined loss using alpha parameter
            distillation_loss = (1 - self.alpha) * weighted_detection_loss + self.alpha * total_cosine_loss
            
            # Calculate current component weights for logging
            current_weights = precision.detach().cpu()
            
            # Safe scalar extraction for logging
            orig_loss_val = original_loss.sum().item()
            weighted_det_loss_val = weighted_detection_loss.item()
            cosine_loss_val = total_cosine_loss.item()
            dist_loss_val = distillation_loss.item()

            # Store current weights for history
            self.component_weight_history['box'].append(current_weights[0].item())
            self.component_weight_history['cls'].append(current_weights[1].item())
            self.component_weight_history['dfl'].append(current_weights[2].item())
            
            # Track metrics
            self.detection_losses.append(weighted_det_loss_val)
            self.cosine_losses.append(cosine_loss_val)
            self.distillation_losses.append(dist_loss_val)

            # Store weight values for monitoring
            for i, layer_idx in enumerate(self.feature_layers):
                self.layer_weight_values[layer_idx] = normalized_weights[i].item()

            return distillation_loss, loss_items
 
        # Set original criterion if not already done
        if not hasattr(trainer.model, "original_criterion"):
            trainer.model.original_criterion = trainer.model.init_criterion()
            
        trainer.model.criterion = distillation_criterion

    def _add_optimizer_param_group(self, trainer, params, name, weight_decay_multiplier=1.0):

        if not hasattr(trainer, 'optimizer') or trainer.optimizer is None:
            return False
            
        # Check if this parameter group already exists
        for group in trainer.optimizer.param_groups:
            if group.get('name') == name:
                LOGGER.info(f"Parameter group '{name}' already exists, skipping addition")
                return False
        
        # Get base parameters from the first parameter group
        pg0 = trainer.optimizer.param_groups[0]
        
        # Get all parameters from base group
        param_values = {k: v for k, v in pg0.items() if k != 'params' and k != 'name'}

        # Override weight decay with multiplier
        if weight_decay_multiplier != 0:
            param_values['weight_decay'] = param_values.get('weight_decay', 0.005) * weight_decay_multiplier
        
        # Create and add the parameter group
        new_param_group = {
            'params': params,
            'name': name,
            **param_values
        }
        
        trainer.optimizer.add_param_group(new_param_group)

        LOGGER.info(f"Added parameter group '{name}' with {len(params) if isinstance(params, list) else sum(1 for _ in params.parameters())} parameters")
        return True
    
    def log_metrics(self, trainer):
        """Clean up hooks and log metrics at the end of each epoch"""
        # Remove all hooks
        for layer_idx, hook in self.teacher_hooks.items():
            hook.remove()
        for layer_idx, hook in self.student_hooks.items():
            hook.remove()

        if len(self.detection_losses) > 0:

            # Log metrics
            avg_detection = sum(self.detection_losses) / max(len(self.detection_losses), 1)
            avg_cosine = sum(self.cosine_losses) / max(len(self.cosine_losses), 1)
            avg_distillation = sum(self.distillation_losses) / max(len(self.distillation_losses), 1)
            
            # Log to wandb with per-layer metrics
            try:
                metrics_dict = {
                    **trainer.metrics,
                    "metrics/fitness": trainer.fitness, 
                    "distillation/detection_loss": avg_detection,
                    "distillation/cosine_loss": avg_cosine, 
                    "distillation/distillation_loss": avg_distillation,
                }
                
                # Add per-layer metrics if available
                for layer_idx in self.feature_layers:
                    layer_prefix = f"distillation/layer_{layer_idx}"

                    adapter_key = f'adapter_{layer_idx}'
                    if adapter_key in self.adapters:
                        # Adapter network - transformation needed to make student features match teacher ones
                        adapter_norm = torch.linalg.norm(next(self.adapters[adapter_key].parameters())).item()
                        metrics_dict[f"{layer_prefix}/adapter_norm"] = adapter_norm
                    #  Contribution weights
                    metrics_dict[f"{layer_prefix}/importance"] = self.layer_weight_values[layer_idx]

                    # Importance weights lr optimizer (only first layer)
                    if hasattr(self, 'layer_weight_optimizer') and layer_idx == 0:
                        # Only log once since it's shared across all layers
                        for pg_idx, pg in enumerate(self.layer_weight_optimizer.param_groups):
                            metrics_dict[f"{layer_prefix}/importance_optimizer"] = pg.get('lr', 0)

                # Add component weights to metrics dictionary
                if len(self.component_weight_history['box']) > 0:
                    # Get the latest weight values
                    metrics_dict["distillation/box_weight"] = self.component_weight_history['box'][-1]
                    metrics_dict["distillation/cls_weight"] = self.component_weight_history['cls'][-1]
                    metrics_dict["distillation/dfl_weight"] = self.component_weight_history['dfl'][-1]

                wandb.log(metrics_dict)

            except Exception as e:
                LOGGER.warning(f"Failed to log metrics to wandb: {e}")

        # Clear metrics for next epoch
        self.detection_losses = []
        self.cosine_losses = []
        self.distillation_losses = []
        self.component_weight_history = {'box': [], 'cls': [], 'dfl': []}
