import typer
from pathlib import Path
from ultralytics import YOLO, settings
import wandb
from loguru import logger
from ray import tune
from ray.tune.tune_config import TuneConfig
from typing import Dict, Optional, Union
import logging

app = typer.Typer()

class EyeBuilder:
    def __init__(self, config: Dict, dataset_path: Union[str, Path]):
        self.config = config
        self.dataset_path = Path(dataset_path)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize model attribute
        self.model = None

    def wandb_init(self, wandb_key: Optional[str] = None) -> None:
        """Setup Weights & Biases tracking."""
        if wandb_key:
            wandb.login(key=wandb_key)
        wandb.init(
            project=self.config["project_name"],
            dir=self.config["wandb"]["dir"]
        )
        settings.update({"wandb": True})

    def initialize_model(self) -> YOLO:
        """Initialize the YOLO model."""
        self.model = YOLO(f"{self.config['model_name']}.pt")
        return self.model

    def train_yolo(self, config, data, device, model_name):
        self.model.train(data=data, device=device, **config)
    
    def train(self, device: str) -> None:
        """Train the YOLO model with specified parameters."""
        if not self.model:
            self.initialize_model()
            
        train_kwargs = self.config["shared_args"] | self.config["train"]
        self.model.train(
            data=str(self.dataset_path),
            device=device,
            **train_kwargs
        )

    def tune(self, device: str) -> tune.ExperimentAnalysis:
        """Perform hyperparameter tuning on the model."""
        if not self.model:
            self.initialize_model()
            
        space = self.config["tune"]["space"]
        tune_kwargs = self.config["shared_args"] | self.config["tune"]["fixed_args"]
        
        self.logger.info(f"Tuning space: {space}")
        self.logger.info(f"Tuning parameters: {tune_kwargs}")
        
        def train_fn(config):
            self.model.train(
                data=str(self.dataset_path),
                device=device,
                **tune_kwargs,
                **config
            )
            
        analysis = tune.run(
            train_fn,
            config=space,
            trial_dirname_creator=lambda trial: trial.trial_id
        )
        
        return analysis