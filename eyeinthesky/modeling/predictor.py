# from pathlib import Path

# from ultralytics.engine.model import Model
# import typer
# from loguru import logger
# from tqdm import tqdm
# from ultralytics import YOLO

# from eyeinthesky.config import CONFIG

# app = typer.Typer()

# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     # features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
#     model_path: Path = CONFIG['models_dir'] / f"{CONFIG['model_name']}.pt",
#     # predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
#     # -----------------------------------------
# ):

#     logger.info("Performing inference for model...")

#     # Load model
#     logger.info(f"Loading model from {model_path}")
#     model = Model()
#     model.load(model_path)
#     # model = YOLO("yolo12n-seg.pt")
#     results = model.predict("path\\to\\file.jpg", save=True)

#     for result in results:
#         # boxes = result.boxes  # Boxes object for bounding box outputs
#         # masks = result.masks  # Masks object for segmentation masks outputs
#         # keypoints = result.keypoints  # Keypoints object for pose outputs
#         # probs = result.probs  # Probs object for classification outputs
#         # obb = result.obb  # Oriented boxes object for OBB outputs
#         result.show()  # display to screen
#         result.save(filename="data\\processed\\result.jpg")  # save to disk
#     logger.success("Inference complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()
