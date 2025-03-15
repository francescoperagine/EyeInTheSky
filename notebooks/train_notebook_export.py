# %%
import os
os.environ['YOLO_VERBOSE'] = 'false'

# %%
%pip install loguru==0.7.3 python-dotenv==1.0.1 PyYAML==6.0.2 torch==2.5.1 tqdm==4.67.1 typer==0.15.1 matplotlib==3.10.0 pyarrow==18.1.0 setuptools==75.1.0 protobuf==4.25.3 ultralytics==8.3.90 ray==2.43.0 albumentations==2.0.5 pandas

# %%
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO, RTDETR, settings
import gc
import json
import locale
import os
import pandas as pd
import sys
import torch
import wandb
import yaml
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import colorstr, LOGGER
import numpy as np

sys.dont_write_bytecode = True
locale.getpreferredencoding = lambda: "UTF-8"

# %%
# Config

config_data = """
wandb:
  project: "EyeInTheSky_merged"
  group: "train"
data: "VisDrone.yaml"
k_samples: 5
train:
  model: "yolo11n.pt"
  project: "EyeInTheSky"
  data: "VisDrone.yaml"
  pretrained: True
  patience: 5
  task: detect
  epochs: 400
  batch: 16
  workers: 8
  seed: 42
  plots: True
  imgsz: 640
  exist_ok: False
  save: True
  save_period: 10
  val: True
  warmup_epochs: 10
  visualize: True
  show: True
  single_cls: False
  rect: False
  resume: False
  fraction: 1.0
  freeze: None
  cache: False
  verbose: False
val:
  project: "EyeInTheSky"
  half: True
  conf: 0.25
  iou: 0.6
  split: "test"
  rect: True
  plots: True
  visualize: True
"""

# %%
# Get device

def get_device() -> str:
    try:
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception as e:
        print(f"Error setting device: {e}")

# %%
# Load config

# config = Config.load("../config/config.yaml")
config = yaml.safe_load(config_data)
config["train"].update({"device" : get_device()})

# %%
# Get Wandb key

def get_wandb_key_colab() -> str:
    try:
        from google.colab import userdata # type: ignore

        if userdata.get("WANDB_API_KEY") is not None:
            return userdata.get("WANDB_API_KEY")
        else:
            raise ValueError("No WANDB key found")
    except:
        return None

def get_wandb_env(path: Path) -> str:
    try:
        from dotenv import dotenv_values # type: ignore

        """Get W&B API key from Colab userdata or environment variable"""

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find .env file at {path}")

        print(f"Loading secrets from {path}")

        secrets = dotenv_values(path)
        print(f"Found keys: {list(secrets.keys())}")

        if "WANDB_API_KEY" not in secrets:
            raise KeyError(f"WANDB_API_KEY not found in {path}. Available keys: {list(secrets.keys())}")

        return secrets['WANDB_API_KEY']
    except:
        return None

def get_wandb_key(path: Path = "../.env") -> str:
    return get_wandb_key_colab() if get_wandb_key_colab() is not None else get_wandb_env(path)

# %%
# Dataset, Trainer, Validator

class VisDroneDataset(YOLODataset):
    """
    Custom dataset for VisDrone that merges pedestrian (0) and people (1) classes.
    Handles class remapping at the earliest possible stage.
    """
    
    # Define the merged names as a class attribute to be accessible from the trainer
    merged_names = {
        0: 'persona',
        1: 'bicicletta',
        2: 'auto',
        3: 'furgone',
        4: 'camion',
        5: 'triciclo',
        6: 'triciclo-tendato',
        7: 'autobus',
        8: 'motociclo'
    }
    
    def __init__(self, *args, **kwargs):
        # Initialize parent class with modified kwargs
        super().__init__(*args, **kwargs)
        
        # Log class mapping
        LOGGER.info(f"{colorstr('VisDroneDataset:')} Using merged classes: {self.merged_names}")
    
    def get_labels(self):
        """
        Load and process labels with class remapping.
        """
        # Get labels from parent method
        labels = super().get_labels()
        
        # Process statistics
        people_count = 0
        shifted_count = 0
        
        # Process labels to merge classes
        for i in range(len(labels)):
            cls = labels[i]['cls']
            
            if len(cls) > 0:
                # Count 'people' instances
                people_mask = cls == 1
                people_count += np.sum(people_mask)
                
                # Merge class 1 (people) into class 0 (pedestrian -> person)
                cls[people_mask] = 0
                
                # Shift classes > 1 down by 1
                gt1_mask = cls > 1
                shifted_count += np.sum(gt1_mask)
                cls[gt1_mask] -= 1
                
                # Store modified labels
                labels[i]['cls'] = cls
        
        # Now set correct class count and names for training
        if hasattr(self, 'data'):
            # Update names and class count
            self.data['names'] = self.merged_names
            self.data['nc'] = len(self.merged_names)
        
        # Log statistics
        person_count = sum(np.sum(label['cls'] == 0) for label in labels)
        LOGGER.info(f"\n{colorstr('VisDroneDataset:')} Remapped {people_count} 'people' instances to {self.merged_names[0]}")
        LOGGER.info(f"{colorstr('VisDroneDataset:')} Total 'persona' instances after merge: {person_count}")
        LOGGER.info(f"{colorstr('VisDroneDataset:')} Shifted {shifted_count} instances of other classes")
        
        return labels

class MergedClassDetectionTrainer(DetectionTrainer):
    """
    Custom trainer that uses VisDroneDataset for merged class training.
    """
    
    def build_dataset(self, img_path, mode="train", batch=None):
        """Build custom VisDroneDataset."""
        return VisDroneDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch or self.batch_size,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect if mode == "train" else True,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls,
            stride=self.stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=None,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )
    
    def set_model_attributes(self):
        """Update model attributes for merged classes."""
        # First call parent method to set standard attributes
        super().set_model_attributes()
        
        # Then update model with the merged class names
        if hasattr(self.model, 'names'):
            # Use the merged names directly from the dataset class
            self.model.names = VisDroneDataset.merged_names
            self.model.nc = len(VisDroneDataset.merged_names)
            
            # Also update data dictionary
            if hasattr(self, 'data'):
                self.data['names'] = VisDroneDataset.merged_names
                self.data['nc'] = len(VisDroneDataset.merged_names)

class MergedClassDetectionValidator(DetectionValidator):
    """
    Custom validator that uses VisDroneDataset for validation/testing with merged classes.
    """
    
    def build_dataset(self, img_path, mode="val", batch=None):
        """Build custom VisDroneDataset for validation."""
        return VisDroneDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch or self.args.batch,
            augment=False,  # no augmentation during validation
            hyp=self.args,
            rect=True,  # rectangular validation for better performance
            cache=None,
            single_cls=self.args.single_cls,
            stride=self.stride,
            pad=0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
        )
    
    def set_model_attributes(self):
        """Update model attributes for merged classes if using a PyTorch model."""
        super().set_model_attributes()
        
        # Update model names if it's a PyTorch model (not for exported models)
        if hasattr(self.model, 'names') and hasattr(self.model, 'model'):
            self.model.names = VisDroneDataset.merged_names
            if hasattr(self.data, 'names'):
                self.data['names'] = VisDroneDataset.merged_names
                self.data['nc'] = len(VisDroneDataset.merged_names)

# %%
# Load top k tune results by fitness

k = config["k_samples"]

csv_path = "../data/processed/wandb_export_2025-03-05T10_24_46.923+01_00.csv"
df = pd.read_csv(csv_path)

df['fitness'] = df['metrics/mAP50(B)'] * 0.1 + df['metrics/mAP50-95(B)'] * 0.9
df = df.dropna(subset=['fitness'])

df_sorted = df.sample(n=k, random_state=42).sort_values(by='fitness', ascending=False)

columns_to_show = ['fitness', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)', 'optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay', 'cos_lr', 'imgsz', 'box', 'cls', 'dfl']

df_k_sampled = df_sorted[columns_to_show].reset_index(drop=True)
print(f"Top {k} Models by Fitness Score:")

display(df_k_sampled)

# %%
# Clear cache

def clear_cache():
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clear Python garbage collector
    gc.collect()

# %%
# Store results

def save_results(dir, name, results):
    os.makedirs(dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"{dir}/{name}_{timestamp}.json"

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    print(f"{name} results saved to {results_path}")

# %%
# Start

def start(model: YOLO | RTDETR, config):
    train_results = model.train(
        trainer=MergedClassDetectionTrainer,
        **config['train']
        )

    test_results = model.val(
        validator=MergedClassDetectionValidator,
        **config['val']
        )

    return train_results, test_results

# %%
# Wandb

settings.update({"wandb": True})

def wandb_start():
    wandb_api_key = get_wandb_key()
    wandb.login(key=wandb_api_key, relogin=True)
    wandb.init(project=config["wandb"]["project"], group=config["wandb"]["group"])
    wandb.log(config["train"])

# %%
model = YOLO(config["train"]["model"])
df_train = df_k_sampled[['optimizer', 'lr0', 'lrf', 'weight_decay', 'box', 'cls', 'dfl']]

# %%
# Train

for idx, trial in df_train.iterrows():
    trial_config = config.copy()
    trial_config["train"].update(trial)

    wandb_start()

    train_results, test_results = start(model, trial_config)

    save_results("../data/processed", "train", train_results)
    save_results("../data/processed", "test", test_results)

    clear_cache()

    wandb.finish()

# %%
# Resume

# resume_config = config.copy()
# del resume_config["val"]["name"]
# resume_config["train"].update({
#     "epochs": 300, 
#     "device": 0,
#     "warmup_epochs": 0,
#     "optimizer": "AdamW",
# })
# resume_config.update(df_train.iloc[0])
# resume_config


