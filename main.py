from eyeinthesky.config import EyeConfig
from eyeinthesky.modeling.train import EyeBuilder
from eyeinthesky.plots import EyePlotter
from pathlib import Path
import os
import sys
import locale
import typer
import sys

sys.dont_write_bytecode = True
locale.getpreferredencoding = lambda: "UTF-8"

app = typer.Typer()

@app.command()
def main():

    project_root = Path(os.path.abspath(os.getcwd()))
    print(f"Project root: {project_root}")

    config_path = project_root / 'config' / 'config.yaml'
    print(f"Config path: {config_path}")
    
    config = EyeConfig.load(config_path)
    device = EyeConfig.get_device()
    print(f"Device: {device}")

    dataset_path = 'VisDrone.yaml'
    builder = EyeBuilder(config=config, dataset_path=dataset_path)

    secrets_path = project_root / ".env"
    print(f"Secrets path: {secrets_path}")

    wandb_api_key = EyeConfig.get_wandb_key(secrets_path)

    builder.wandb_init(wandb_key=wandb_api_key, project_root=project_root)
    
    analysis = builder.tune(device=device)
    print(f"Analysis: {analysis}")

    plotter = EyePlotter()
    plotter.show_trial_results_metrics(analysis)
    plotter.show_results_plots(analysis, config["reports_dir"], config["name"])

if __name__ == "__main__":
    app()