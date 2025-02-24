import yaml
from pathlib import Path
import os
from dotenv import dotenv_values
import torch
from ray import tune

class EyeConfig:
    """Singleton class for managing project configuration and secrets."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    def load(config_file: str) -> dict:
        """Load and return configuration from YAML file."""
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
        
    @staticmethod
    def get_space(config) -> dict:
        """Convert config space parameters to tune.uniform objects"""
        space = {}
        for param, value in config.items():
            if isinstance(value, dict):  # It's a min/max range
                space[param] = tune.uniform(value["min"], value["max"])
            else:  # It's a discrete choice list
                space[param] = tune.choice(value)
        return space
    
    @staticmethod
    def get_device() -> str:
        try:
            return 0 if torch.cuda.is_available() else "cpu"
        except Exception as e:
            print(f"Error setting device: {e}")

    @staticmethod
    def get_wandb_key() -> str:
        """Get W&B API key from Colab userdata or environment variable"""
        try:
            # Try to import google.colab (will raise ImportError if not in Colab)
            from google.colab import userdata
            if userdata.get("WANDB_API_KEY") is not None:
                return userdata.get("WANDB_API_KEY")
            else:
                raise ValueError("No WANDB key found")
        except:
            # Not in Colab, use environment variable instead
            from dotenv import dotenv_values
            secrets = dotenv_values(".env")
            return secrets['WANDB_API_KEY']