import typer
from pathlib import Path
from ultralytics import YOLO, settings
import wandb
from typing import Dict, Optional, Union
import logging
import os

app = typer.Typer()

class EyeBuilder:
    def __init__(self, config: Dict, dataset_path, model: Optional[YOLO] = None) -> None:
        self.config = config
        self.dataset_path = dataset_path
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize model attribute
        self.model = model if model else YOLO(f"{self.config['model_name']}.pt")

    def wandb_init(self, wandb_key: Optional[str] = None, project_root: Optional[Union[str, Path]] = None) -> None:
        """Setup Weights & Biases tracking.
        
        Args:
            wandb_key: Optional API key for Weights & Biases
            project_root: Optional path to the project root directory. If not provided,
                        will use the current working directory.
        """
        # Use provided project root or fall back to current directory
        if project_root is None:
            project_root = Path(os.getcwd())
        else:
            project_root = Path(project_root)
        
        # Create the full path for wandb directory
        wandb_dir = project_root / self.config["wandb"]["dir"]
        self.logger.info(f"Using wandb directory: {wandb_dir}")
        
        # Create directory if it doesn't exist
        wandb_dir.mkdir(parents=True, exist_ok=True)
        
        if wandb_key:
            wandb.login(key=wandb_key)
        
        wandb.init(
            project=self.config["project_name"],
            dir=str(wandb_dir)
        )
        
        settings.update({"wandb": True})
    
    def train(self, device: str) -> None:
        """Train the YOLO model with specified parameters."""
            
        train_kwargs = self.config["shared_args"] | self.config["train"]
        self.model.train(
            data=str(self.dataset_path),
            device=device,
            **train_kwargs
        )

    def tune(self, device: str):
        """Perform hyperparameter tuning on the model."""

        search_space = {
            "lr0": (1e-5, 1e-3),     # Keep it low for fine-tuning
            "lrf": (0.01, 0.1),         # Learning rate factor
            "momentum": (0.9, 0.95),         # High momentum for stability
            "weight_decay": (0.0, 0.001),         # Minimal regularization
            "box": (1.0, 20.0),  # box loss gain
            "cls": (0.2, 4.0),  # cls loss gain (scale with pixels)
            "dfl": (0.4, 6.0),  # dfl loss gain
            "scale": (0.0, 0.95),  # image scale (+/- gain)
            "degrees": (0.0, 45.0),  # image rotation (+/- deg)
        }

            # "warmup_epochs": (0.0, 5.0),  # warmup epochs (fractions ok)
            # "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
            # "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            # "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            # "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
            # "translate": (0.0, 0.9),  # image translation (+/- fraction)
            # "shear": (0.0, 10.0),  # image shear (+/- deg)
            # "perspective": (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            # "flipud": (0.0, 1.0),  # image flip up-down (probability)
            # "fliplr": (0.0, 1.0),  # image flip left-right (probability)
            # "bgr": (0.0, 1.0),  # image channel bgr (probability)
            # "mosaic": (0.0, 1.0),  # image mixup (probability)
            # "mixup": (0.0, 1.0),  # image mixup (probability)
            # "copy_paste": (0.0, 1.0),  # segment copy-paste (probability)

        print(f"Space: {search_space}")

        result_grid = self.model.tune(
            data=str(self.dataset_path),
            device=device,
            epochs=self.config["tune"]["epochs"],
            batch_size=self.config["tune"]["batch_size"],
            iterations=self.config["tune"]["iterations"],
            workers=self.config["tune"]["workers"],
            seed=self.config["tune"]["seed"],
            plots=self.config["tune"]["plots"],
            val=self.config["tune"]["val"],
            cos_lr=self.config["tune"]["cos_lr"],
            use_ray=self.config["tune"]["use_ray"],
            imgsz=self.config["tune"]["imgsz"],
            exist_ok=self.config["tune"]["exist_ok"],
            save=self.config["tune"]["save"],
            save_period=self.config["tune"]["save_period"],
            space=search_space,
        )
        return result_grid

    # def tune(self, device: str) -> tune.ExperimentAnalysis:
    #     """Perform hyperparameter tuning on the model."""
    #     if not self.model:
    #         self.initialize_model()
            
    #     raw_space = self.config["tune"]["space"]
    #     space = {}
    #     for key, value in raw_space.items():
    #         if isinstance(value, list):
    #             space[key] = tune.choice(value)
    #         else:
    #             space[key] = value
    #     print(f"Space: {space}")
                
    #     tune_kwargs = self.config["shared_args"] | self.config["tune"]["fixed_args"]
        
    #     self.logger.info(f"Tuning space: {space}")
    #     self.logger.info(f"Tuning parameters: {tune_kwargs}")
        
    #     def train_fn(config):
    #         self.model.train(
    #             data=str(self.dataset_path),
    #             device=device,
    #             **tune_kwargs,
    #             **config
    #         )

    #     def short_dirname_creator(trial):
    #         return f"trial_{trial.trial_id[-8:]}"
        
    #     resources_per_trial = {"cpu": 1}
    #     if "cuda" in device:
    #         resources_per_trial["gpu"] = 1
                
    #     analysis = tune.run(
    #         train_fn,
    #         config=space,
    #         resources_per_trial=resources_per_trial,
    #         trial_dirname_creator=short_dirname_creator
    #     )
        
    #     return analysis

    # def tune(self, device: str) -> tune.ExperimentAnalysis:
    #     """Perform hyperparameter tuning on the model."""
    #     if not self.model:
    #         self.initialize_model()
            
    #     # Get the raw space config from YAML
    #     raw_space = self.config["tune"]["space"]
        
    #     # Dynamically convert lists to tune.choice
    #     space = {}
    #     for key, value in raw_space.items():
    #         if isinstance(value, list):
    #             space[key] = tune.choice(value)
    #         else:
    #             space[key] = value
        
    #     tune_kwargs = self.config["shared_args"] | self.config["tune"]["fixed_args"]
        
    #     self.logger.info(f"Tuning space: {space}")
    #     self.logger.info(f"Tuning parameters: {tune_kwargs}")
        
    #     # Get reports directory from config
    #     reports_dir = Path(self.config["reports_dir"])
    #     # Create the directory if it doesn't exist
    #     reports_dir.mkdir(parents=True, exist_ok=True)
        
    #     # Create a logs subdirectory for TensorBoard
    #     tensorboard_dir = reports_dir / "tensorboard_logs"
    #     tensorboard_dir.mkdir(exist_ok=True)
        
    #     # Create a temp dir for Ray Tune storage
    #     tune_dir = reports_dir / "ray_tune"
    #     tune_dir.mkdir(exist_ok=True)
        
    #     def train_fn(config):
    #         self.model.train(
    #             data=str(self.dataset_path),
    #             device=device,
    #             **tune_kwargs,
    #             **config
    #         )
        
    #     # Create a custom function to generate shorter trial directory names
    #     def short_dirname_creator(trial):
    #         """Create a short directory name for each trial to avoid Windows path length issues."""
    #         return f"trial_{trial.trial_id[-8:]}"
        
    #     # Add resource constraints to prevent overloading the system
    #     resources_per_trial = {"cpu": 1}
    #     if "cuda" in device:
    #         resources_per_trial["gpu"] = 1
        
    #     # Configure TensorBoard callback
    #     tensorboard_callback = tune.logger.TBXLoggerCallback(
    #         logdir=str(tensorboard_dir)
    #     )
        
    #     # Run the tuning process
    #     analysis = tune.run(
    #         train_fn,
    #         config=space,
    #         resources_per_trial=resources_per_trial,
    #         trial_dirname_creator=short_dirname_creator,
    #         local_dir=str(tune_dir),  # Set the local directory for ray tune results
    #         callbacks=[tensorboard_callback],  # Add TensorBoard callback
    #         name=self.config["project_name"]
    #     )
        
    #     # Print the location of the TensorBoard logs
    #     print(f"\nTo visualize your results with TensorBoard, run: `tensorboard --logdir {tensorboard_dir}`\n")
        
    #     return analysis

    # def tune(self, device: str) -> tune.ExperimentAnalysis:
    #     """Simplified hyperparameter tuning approach."""
    #     if not self.model:
    #         self.initialize_model()
        
    #     # Simplify to just two key hyperparameters
    #     space = {
    #         "lr0": tune.choice([0.0001, 0.001]),
    #         "cls": tune.choice([0.5, 1.0])
    #     }
        
    #     # Get the fixed arguments from config
    #     tune_kwargs = self.config["shared_args"] | self.config["tune"]["fixed_args"]
        
    #     # Remove potentially problematic parameters
    #     for key in ["iterations", "use_ray", "grace_period"]:
    #         if key in tune_kwargs:
    #             tune_kwargs.pop(key)
        
    #     # Set fixed values for other hyperparameters that were previously tunable
    #     tune_kwargs["optimizer"] = "adamw"
    #     tune_kwargs["momentum"] = 0.9
    #     tune_kwargs["weight_decay"] = 0.0
        
    #     # Basic training function
    #     def train_fn(config):
    #         self.model.train(
    #             data=str(self.dataset_path),
    #             device=device,
    #             **tune_kwargs,
    #             **config
    #         )
        
    #     # Simple directory creator
    #     def simple_dirname_creator(trial):
    #         return f"trial_{trial.trial_id[-6:]}"
        
    #     # Create the reports directory for storage
    #     reports_dir = Path(self.config["reports_dir"])
    #     reports_dir.mkdir(parents=True, exist_ok=True)
    #     ray_dir = reports_dir / "ray_tune"
    #     ray_dir.mkdir(exist_ok=True)

    #     absolute_path = str(ray_dir.absolute())
    #     self.logger.info(f"Using Ray Tune storage path: {absolute_path}")
        
    #     # Run with minimal options
    #     analysis = tune.run(
    #         train_fn,
    #         config=space,
    #         resources_per_trial={"cpu": 1, "gpu": 0.5 if "cuda" in device else 0},
    #         trial_dirname_creator=simple_dirname_creator,
    #         storage_path=absolute_path,
    #         name="simple_tune",
    #         num_samples=1,  # Start with just 1 sample of each parameter combination
    #         verbose=1
    #     )
        
    #     return analysis

    # def tune(self, device: str) -> tune.ExperimentAnalysis:
    #     """Optimized hyperparameter tuning approach."""
    #     if not self.model:
    #         self.initialize_model()
        
    #     # Just tune learning rate and cls for simplicity
    #     space = {
    #         "lr0": tune.choice([0.0001, 0.001]),
    #         "cls": tune.choice([0.5, 1.0])
    #     }
        
    #     # Get the fixed arguments from config
    #     tune_kwargs = self.config["shared_args"] | self.config["tune"]["fixed_args"]
        
    #     # Remove Ray-specific parameters that might conflict
    #     keys_to_remove = ["iterations", "use_ray", "grace_period"]
    #     for key in keys_to_remove:
    #         if key in tune_kwargs:
    #             tune_kwargs.pop(key)
        
    #     # Fix all other previously tunable parameters
    #     tune_kwargs["optimizer"] = "adamw"
    #     tune_kwargs["momentum"] = 0.9
    #     tune_kwargs["weight_decay"] = 0.0
        
    #     # Store large objects in Ray's object store to prevent copying
        
    #     # dataset_path_ref = ray.put(str(self.dataset_path))
    #     # tune_kwargs_ref = ray.put(tune_kwargs)
        
    #     # Basic training function that references objects from the store
    #     def train_fn(config):
    #         # Get dataset_path and tune_kwargs from Ray's object store
    #         dataset_path = self.dataset_path
    #         kwargs = tune_kwargs
            
    #         # # Re-initialize model in each worker to avoid serializing it
    #         # from ultralytics import YOLO
    #         # model = YOLO(f"{self.config['model_name']}.pt")
            
    #         # Train with the configuration
    #         self.model.train(
    #             data=dataset_path,
    #             device=device,
    #             **kwargs,
    #             **config
    #         )
        
    #     # Run with minimal options
    #     analysis = tune.run(
    #         train_fn,
    #         config=space,
    #         resources_per_trial={"cpu": 1},
    #         num_samples=1,
    #         verbose=1
    #     )
        
    #     return analysis    