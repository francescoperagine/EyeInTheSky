import typer
from ultralytics import YOLO
from typing import Optional
from .builder import Builder

app = typer.Typer()

class Tuner(Builder):
    """Class responsible for hyperparameter tuning of YOLO models."""
    
    def tune(self, search_space):
        """Perform hyperparameter tuning on the model."""
        try:
            self.wandb_init("tune")
            
            tune_config = self.config["tune"]
            result_grid = self.model.tune(
                data=str(self.data),
                device=self.device,
                gpu_per_trial=1,
                project=self.config["project"],
                name=tune_config["name"],
                epochs=tune_config["epochs"],
                iterations=tune_config["iterations"],
                batch=tune_config["batch"],
                workers=tune_config["workers"],
                seed=tune_config["seed"],
                plots=tune_config["plots"],
                val=tune_config["val"],
                cos_lr=tune_config["cos_lr"],
                use_ray=tune_config["use_ray"],
                imgsz=tune_config["imgsz"],
                exist_ok=tune_config["exist_ok"],
                save=tune_config["save"],
                save_period=tune_config["save_period"],
                space=search_space,
            )
            return result_grid
        finally:
            self.wandb_cleanup()


@app.command()
def tune(
    model_path: str = typer.Argument("models/yolov12n.pt", help="Path to the base YOLO model"),
    config_path: str = typer.Argument("config/config.yaml", help="Path to configuration file"),
    dataset_path: str = typer.Argument("config/VisDrone.yaml", help="Path to dataset"),
    search_space_path: str = typer.Argument("config/search_space.json", help="Path to search space JSON file"),
    wandb_key: Optional[str] = typer.Option(None, help="Weights & Biases API key"),
    project_root: Optional[str] = typer.Option(None, help="Project root directory")
):
    """Tune hyperparameters for a YOLO model with the specified configuration."""
    import yaml
    import json
    from ray import tune
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and parse search space
    with open(search_space_path, 'r') as f:
        search_space_json = json.load(f)
    
    # Convert JSON to Ray Tune objects
    search_space = {}
    for param, config in search_space_json.items():
        if config["type"] == "choice":
            search_space[param] = tune.choice(config["values"])
        elif config["type"] == "uniform":
            search_space[param] = tune.uniform(config["min"], config["max"])
    
    # Load model
    model = YOLO(model_path)
    
    # Initialize tuner
    tuner = Tuner(
        model=model,
        config=config,
        data=dataset_path,
        wandb_key=wandb_key,
        project_root=project_root
    )
    
    # Tune model
    results = tuner.tune(search_space)
    print(f"Tuning completed. Best parameters: {results['best_params']}")


if __name__ == "__main__":
    app()