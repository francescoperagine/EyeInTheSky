from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from ray.tune import ResultGrid

def show_trial_results_metrics(result_grid: ResultGrid):
    for i, result in enumerate(result_grid):
        print(f"Trial #{i}: Configuration: {result.config}, Last Reported Metrics: {result.metrics}")

def show_results_plots(result_grid: ResultGrid, destination_folder: str, experiment_name: str):
    for i, result in enumerate(result_grid):
        plt.plot(
        result.metrics_dataframe["training_iteration"],
        result.metrics_dataframe["mean_accuracy"],
        label=f"Trial {i}",
    )

    plt.xlabel("Training Iterations")
    plt.ylabel("Mean Accuracy")
    plt.legend()
    plt.show()
    plt.savefig(destination_folder / f"{experiment_name}.png")     