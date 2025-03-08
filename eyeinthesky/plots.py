import matplotlib.pyplot as plt

class Plotter:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def show_trial_results_metrics(result_grid):
        for i, result in enumerate(result_grid):
            print(f"Trial #{i}: Configuration: {result.config}, Last Reported Metrics: {result.metrics}")

    @staticmethod
    def show_results_plots(result_grid, destination_folder: str, experiment_name: str):
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