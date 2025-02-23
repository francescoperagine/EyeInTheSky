import typer
from pathlib import Path
from ultralytics import YOLO, checks, settings
from eyeinthesky.config import ProjectConfig
import wandb

from loguru import logger
import os

app = typer.Typer()

@app.command()
def main():

    PROJECT_ROOT = Path(os.getcwd())

    config_path = os.path.join(PROJECT_ROOT, "config")
    config_file = os.path.join(config_path, "config.yaml")
    config = ProjectConfig.get_config(str(config_file))
    
    dataset_file = f"{Path(config_path) / config['dataset_name']}.yaml"

    wandb_key = ProjectConfig.get_wandb_key()
    device = ProjectConfig.get_device()

    wandb.login(key=wandb_key)
    wandb.init(project=config["project_name"], dir=config["wandb"]["dir"])
    settings.update({"wandb": True})

    logger.info(f"Performing training for model {config['model_name']}...")
    logger.info(checks())

    kwargs = config["shared_args"] | config["train"]
    
    model = YOLO(f"{config['model_name']}.pt")

    train_results = model.train(data=dataset_file, device=device, **kwargs)

if __name__ == "__main__":
    app()
