import torch
from pathlib import Path
import os
import gc
import yaml 
import glob

def load_config(config_file: str) -> dict:
    """Load and return configuration from YAML file."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def _get_wandb_key_colab() -> str:
    try:
        from google.colab import userdata # type: ignore

        if userdata.get("WANDB_API_KEY") is not None:
            return userdata.get("WANDB_API_KEY")
        else:
            raise ValueError("No WANDB key found")
    except:
        return None

def _get_wandb_env(path: Path) -> str:
    try:
        from dotenv import dotenv_values # type: ignore

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
    except:
        return None

def get_wandb_key(path: Path = "../.env") -> str:
    return _get_wandb_key_colab() if _get_wandb_key_colab() is not None else _get_wandb_env(path)

def clear_cache():
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clear Python garbage collector
    gc.collect()

def get_device() -> str:
    try:
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception as e:
        print(f"Error setting device: {e}")

def remove_models():
    pt_files = glob.glob("*.pt")
    print("Files to be removed:", pt_files)

    for file in pt_files:
        os.remove(file)