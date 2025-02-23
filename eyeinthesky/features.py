import typer
from pathlib import Path
from ultralytics import YOLO, checks, settings
from eyeinthesky.config import ProjectConfig
import wandb
from loguru import logger
import os
import plots
from ray import tune
from ray.tune.tune_config import TuneConfig

def custom_trial_dirname_creator(trial):
    return trial.trial_id
# def main():
#     PROJECT_ROOT = Path(os.getcwd())
#     config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")

#     config = ProjectConfig.get_config(str(config_path))
#     wandb_key = ProjectConfig.get_wandb_key()
#     device = ProjectConfig.get_device()

#     settings.update({"wandb": True})

#     wandb.login(key=wandb_key)
#     # wandb.init(project=config["project_name"], dir=config["wandb"]["dir"])

#     logger.info(f"Performing tuning for model {config['model_name']}...")
#     logger.info(checks())
    
#     model = YOLO(f"{config['model_name']}.pt")

#     # Extract configuration
#     tune_args = config["tune"]
#     space = ProjectConfig.get_space_dict(tune_args["space"])
#     kwargs = tune_args["fixed_args"]
#     print(kwargs)

#     result_grid = model.tune(data=f"{config['dataset_name']}.yaml", device=device, space=space, **kwargs)
    
#     plots.show_trial_results_metrics(result_grid)
#     plots.show_results_plots(result_grid, config["name"])


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

    logger.info(f"Performing tuning for model {config['model_name']}...")
    logger.info(checks())
    
    tune_kwargs = config["shared_args"] | config["tune"]["fixed_args"]
    train_kwargs = config["shared_args"] | config["train"]

    space = ProjectConfig.get_space_dict(config["tune"]["space"])
    print("Tuning space:", space)
    print("Tuning parameters:", tune_kwargs)

    model = YOLO(f"{config['model_name']}.pt")

    tune_config = tune.TuneConfig(
        trial_dirname_creator=lambda trial: trial.trial_id,
    )

    # result_grid = model.tune(data=dataset_file, 
    #     device=device, 
    #     space=space, 
    #     tune_config=tune_config,
    #     **kwargs
    # )

    def train_yolo(config, data, device, model_name):
    # model = YOLO(f"{model_name}.pt")
        model.train(data=data, device=device, **train_kwargs)

    analysis = tune.run(
        tune.with_parameters(train_yolo,
            data=dataset_file, 
            device=device, 
            model_name=config['model_name']),
        config=space,
        trial_dirname_creator=lambda trial: trial.trial_id,
    )
                          
    plots.show_trial_results_metrics(analysis)
    plots.show_results_plots(analysis, config["reports_dir"], config["name"])

if __name__ == "__main__":
    app()