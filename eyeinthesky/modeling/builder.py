from pathlib import Path
from ultralytics import YOLO, settings
import wandb
from typing import Dict, Optional, Union
import logging
import os
from eyeinthesky.config import Config

class Builder:
    """Base class containing common functionality for model operations.
    
    This class serves as a foundation for specialized model builders (Trainer, Tuner, etc.) 
    in the EyeInTheSky project, providing shared functionality for model initialization,
    weights & biases integration, and resource cleanup.
      
    Args:
        model (YOLO): Initialized YOLO model instance.
        config (Dict): Configuration dictionary containing operation parameters.
        dataset_path: Path to the dataset configuration YAML or dataset directory.
        wandb_key (Optional[str], optional): Weights & Biases API key. Defaults to None.
        project_root (Optional[Union[str, Path]], optional): Project root directory. 
            If None, current working directory is used. Defaults to None.
            
    Example:
        ```python
        from ultralytics import YOLO
        from eyeinthesky.config import Config
        import yaml
        
        # Load config
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        # Initialize model
        model = YOLO("models/yolov12n.pt")
        
        # Create builder
        builder = Builder(
            model=model,
            config=config,
            dataset_path="config/VisDrone.yaml",
            wandb_key=wandb_api_key,
            project_root="path/to/project"
        )
        ```
    """
    
    def __init__(self, model: str, config: Dict, data, wandb_key: Optional[str] = None, 
                 project_root: Optional[Union[str, Path]] = None) -> None:
        self.model = model
        self.config = config
        self.data = data
        self.wandb_key = wandb_key
        self.project_root = project_root

        self.logger = logging.getLogger(__name__)
        self.device = Config.get_device()

    def wandb_init(self, operation_name: str) -> None:
        """Setup Weights & Biases tracking.

        Args:
            operation_name: Name of the operation (training/tuning) for the W&B run
        """
        # Use provided project root or fall back to current directory
        if self.project_root is None:
            self.project_root = Path(os.getcwd())

        # Create the full path for wandb directory
        wandb_dir = self.project_root / self.config["wandb"]["dir"]
        self.logger.info(f"Using wandb directory: {wandb_dir}")

        # Create directory if it doesn't exist
        wandb_dir.mkdir(parents=True, exist_ok=True)

        if self.wandb_key:
            wandb.login(key=self.wandb_key, relogin=True)

        wandb.init(
            project=self.config["project"],
            name=f"{self.config['model_name']}_{self.config['dataset_name']}_{operation_name}",
            dir=str(wandb_dir),
        )

        settings.update({"wandb": True})
        
    def wandb_cleanup(self) -> None:
        """Cleanup resources after operation is complete."""
        wandb.finish()