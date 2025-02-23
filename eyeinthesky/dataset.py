import os
import shutil
import typer
from loguru import logger
from eyeinthesky.config import CONFIG, ROBOFLOW_API_KEY
from roboflow import Roboflow
from eyeinthesky.dataset import CarlaSimulator, SimulationConfig

app = typer.Typer()

def get_datasets(api_key: str, destination_folder: str) -> dict:
    logger.info("Downloading datasets...")

    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)

    # Define the datasets you want to download
    datasets = [
        {
            'name': 'car_seg_un1pm',
            'workspace': 'gianmarco-russo-vt9xr',
            'project': 'car-seg-un1pm',
            'version': 4,
            'model': 'yolov11'
        },
        {
            'name': 'accident_detection_k2qd0',
            'workspace': 'rd-vlvvq',
            'project': 'accident-detection-k2qd0',
            'version': 1,
            'model': 'yolov11'
        },
        {
            'name': 'car_fire_yssjr',
            'workspace': 'traffic-ai-8xnmy',
            'project': 'car-fire-yssjr',
            'version': 1,
            'model': 'yolov11'
        },
        {
            'name': 'accident_detection_model',
            'workspace': 'firsttest-eeat7',
            'project': 'accident_detection_model',
            'version': 5,
            'model': 'yolov11'
        },
        {
            'name': 'accident_detection_77mha',
            'workspace': 'yolo-and-car-accident-detection-xaltb',
            'project': 'accident-detection-77mha',
            'version': 2,
            'model': 'yolov11'
        },
    ]

    dataset_paths = {}

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    for ds in datasets:
        logger.info(f"Downloading dataset: {ds['name']} ...")
        # Access the correct workspace and project, then select the version.
        project = rf.workspace(ds['workspace']).project(ds['project'])
        version_obj = project.version(ds['version'])
        
        # Download the dataset
        ds_obj = version_obj.download(ds['model'])
        src_folder = ds_obj.location  # The folder where the dataset was initially downloaded.

        # Define a new destination sub-folder for this dataset
        dest_subfolder = os.path.join(destination_folder, ds['workspace'])
        
        # If the subfolder exists, remove it to avoid conflicts.
        if os.path.exists(dest_subfolder):
            shutil.rmtree(dest_subfolder)
        
        # Move the downloaded folder to the destination subfolder.
        shutil.move(src_folder, dest_subfolder)
        logger.info(f"Dataset {ds['name']} downloaded to {dest_subfolder}")
        
        dataset_paths[ds['name']] = dest_subfolder

    return dataset_paths

@app.command()
def main():
    logger.info("Processing datasets...")
    dataset_paths = get_datasets(api_key=ROBOFLOW_API_KEY, destination_folder=CONFIG['raw_data_dir'])
    logger.info(f"Downloaded datasets: {dataset_paths}")

    # config = SimulationConfig(num_scenes=100)

    # with CarlaSimulator(config) as simulator:
    #     simulator.generate_dataset()  # Container starts

if __name__ == "__main__":
    app()
