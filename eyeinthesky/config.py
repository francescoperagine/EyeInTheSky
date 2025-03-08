import yaml
from pathlib import Path
import os
from dotenv import dotenv_values
import torch

class Config:
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
    def get_device() -> str:
        try:
            return 0 if torch.cuda.is_available() else "cpu"
        except Exception as e:
            print(f"Error setting device: {e}")

    @staticmethod
    def get_wandb_key_colab() -> str:
        from google.colab import userdata # type: ignore
        if userdata.get("WANDB_API_KEY") is not None:
            return userdata.get("WANDB_API_KEY")
        else:
            raise ValueError("No WANDB key found")
    @staticmethod
    def get_wandb_key(path: Path = ".env") -> str:
        """Get W&B API key from Colab userdata or environment variable"""

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find .env file at {path}")

        print(f"Loading secrets from {path}")

        secrets = dotenv_values(path)
        print(f"Found keys: {list(secrets.keys())}")

        if "WANDB_API_KEY" not in secrets:
            raise KeyError(f"WANDB_API_KEY not found in {path}. Available keys: {list(secrets.keys())}")

        return secrets['WANDB_API_KEY']
    
def main():
    config_path = Path(os.getcwd()) / 'config' / 'config.yaml'
    print(f"Loading config from {config_path}")
    config = Config.load(config_path)
    device = Config.get_device()

    dataset_path = Path(os.path.abspath(os.path.join(os.getcwd(), '..'))) / 'config' / 'VisDrone.yaml'


    wandb_key = Config.get_wandb_key()

    print(wandb_key)
