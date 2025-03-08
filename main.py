from eyeinthesky.config import Config
from eyeinthesky.modeling.trainer import Trainer
from eyeinthesky.modeling.tuner import Tuner
from pathlib import Path
import os
import sys
import locale
import typer
import sys
from ultralytics import YOLO

sys.dont_write_bytecode = True
locale.getpreferredencoding = lambda: "UTF-8"

app = typer.Typer()

@app.command()
def main():

    # subprocess.check_call([sys.executable, "-m", "venv", "venv"])
    # subprocess.check_call(["source", "venv/bin/activate"])
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])

    project_root = Path(os.path.abspath(os.getcwd()))
    print(f"Project root: {project_root}")

    config_path = project_root / 'config' / 'config.yaml'
    print(f"Config path: {config_path}")
    
    config = Config.load(config_path)
    device = Config.get_device()
    print(f"Device: {device}")

    dataset_path = 'VisDrone.yaml'

    secrets_path = project_root / ".env"
    print(f"Secrets path: {secrets_path}")

    wandb_api_key = Config.get_wandb_key(secrets_path)

    model = YOLO(f"{config['model_name']}.pt")

    # For tuning
    tuner = Tuner(model, config, dataset_path, wandb_api_key, project_root)
    tuning_results = tuner.tune("config/search_space.json")

    # For training
    trainer = Trainer(model, config, dataset_path, wandb_api_key, project_root)
    training_results = trainer.train()

if __name__ == "__main__":
    app()