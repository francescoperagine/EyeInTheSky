from eyeinthesky.config import EyeConfig
from eyeinthesky.modeling.train import EyeBuilder
from eyeinthesky.plots import EyePlotter
from pathlib import Path
import os
import sys
import locale
import typer

sys.dont_write_bytecode = True
locale.getpreferredencoding = lambda: "UTF-8"

app = typer.Typer()


@app.command()
def main():
    config_path = Path(os.getcwd()) / 'config' / 'config.yaml'
    print(f"Loading config from {config_path}")
    config = EyeConfig.load(config_path)
    device = EyeConfig.get_device()

    dataset_path = Path(os.path.abspath(os.path.join(os.getcwd(), '..'))) / 'config' / 'VisDrone.yaml'

    builder = EyeBuilder(config=config, dataset_path=dataset_path)

    wandb_key = EyeConfig.get_wandb_key()

    builder.wandb_init(wandb_key)

    analysis = builder.tune(device=device)

    plotter = EyePlotter()
    plotter.show_trial_results_metrics(analysis)
    plotter.show_results_plots(analysis, config["reports_dir"], config["name"])


if __name__ == "__main__":
    app()