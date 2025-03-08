from .builder import Builder 
from eyeinthesky.dataset import MergedDataset
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.torch_utils import torch_distributed_zero_first

class CustomDetectionTrainer(DetectionTrainer):

    def __init__(self, config=DEFAULT_CFG, overrides=None, _callbacks=None, merge_config=None):
        self.merge_config = merge_config

        super().__init__(cfg=config, overrides=overrides, _callbacks=_callbacks)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            if self.merge_config is not None:
                LOGGER.warning(f"Merge mapping found: {self.merge_config}")
                dataset = MergedDataset(original_dataset=dataset, merge_mapping=self.merge_config)
            else:
                LOGGER.warning("Merge mapping not found")
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

class Trainer(Builder):
    """Class responsible for training YOLO models."""
    
    def train(self):
        """Train the YOLO model with specified parameters."""
        # try:
            # self.wandb_init("train")
            
        # train_kwargs = self.config["train"]
        # results = self.model.train(
        #     model = 'yolo12n.pt',
        #     data=str(self.dataset_path),
        #     device=self.device,
        #     **train_kwargs
        # )
        # return results
    
        args = dict(model=self.model,
            data=self.data,
            device=self.device
        )
        LOGGER.info(f"Config: {self.config['train']}\nOverride args: {args}\nMerge config: {self.config['merge_mapping']}")
        
        trainer = CustomDetectionTrainer(config=self.config["train"], overrides=args, merge_config=self.config['merge_mapping'])
        
        results = trainer.train()
        return results
    # finally:
            # self.wandb_cleanup()


# def train(self):
#         """Train the YOLO model with specified parameters.
        
#         If 'merge_mapping' exists in config, the YOLOv12 dataset loader is monkey-patched
#         to wrap the original dataset with MergedDataset, which remaps labels on the fly.
#         """
#         try:
#             self.wandb_init("train")
#             patch_applied = False

#             # If merge_mapping is defined in the configuration, apply the monkey-patch.
#             if "merge_mapping" in self.config:
#                 self.logger.info("Applying merge mapping: " + str(self.config["merge_mapping"]))
#                 original_get_dataset = yolo_data.get_dataset  # assume this is the function used internally
#                 merge_mapping = self.config["merge_mapping"]

#                 def custom_get_dataset(*args, **kwargs):
#                     ds = original_get_dataset(*args, **kwargs)
#                     return CustomDataset(ds, merge_mapping)

#                 yolo_data.get_dataset = custom_get_dataset
#                 patch_applied = True

#             train_kwargs = self.config["train"]
#             results = self.model.train(
#                 data=str(self.dataset_path),
#                 device=self.device,
#                 **train_kwargs
#             )
#             return results
#         finally:
#             if patch_applied:
#                 # Revert the monkey-patch to avoid side effects in future runs.
                
#                 yolo_data.get_dataset = original_get_dataset
#             self.cleanup()



# from ultralytics import YOLO
# from typing import Optional
# from .build import Builder
# from .data import MergedDataset
# from ultralytics.models.yolo.detect.train import DetectionTrainer
# import ultralytics.data as yolo_data


        
# class Trainer(Builder):
#     """
#     Class responsible for training YOLO models using a customized DetectionTrainer API.
    
#     This Trainer supports on-the-fly class merging via a custom dataset wrapper,
#     and uses CustomDetectionTrainer to override the model instantiation.
#     """
    
#     def train(self):
#         try:
#             # self.wandb_init("train")
#             patch_applied = False

#             # Apply dataset merge mapping if specified in the config.
#             if "merge_mapping" in self.config:
#                 self.logger.info("Applying merge mapping: " + str(self.config["merge_mapping"]))
#                 original_build_yolo_dataset = yolo_data.build_yolo_dataset
#                 merge_mapping = self.config["merge_mapping"]

#                 def custom_build_yolo_dataset(*args, **kwargs):
#                     ds = original_build_yolo_dataset(*args, **kwargs)
#                     return MergedDataset(ds, merge_mapping)

#                 yolo_data.build_yolo_dataset = custom_build_yolo_dataset
#                 patch_applied = True

#                 args = dict(model=self.model,
#                     data=self.dataset_path,
#                     device=self.device,
#                     **self.config["train"]
#                 )

#             trainer = DetectionTrainer(overrides=args)
            
#             results = trainer.train()
#             return results
#         finally:
#             if patch_applied:
#                 yolo_data.build_yolo_dataset = original_build_yolo_dataset
#             self.cleanup()