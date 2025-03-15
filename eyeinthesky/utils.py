import torch
import datetime
from pathlib import Path
import os
import gc
import json
import yaml 


def load(config_file: str) -> dict:
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

def save_results(dir, name, results):
    os.makedirs(dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"{dir}/{name}_{timestamp}.json"

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    print(f"{name} results saved to {results_path}")

def get_device() -> str:
    try:
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception as e:
        print(f"Error setting device: {e}")