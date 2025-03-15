from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from eyeinthesky.dataset import VisDroneMergedDataset

class MergedClassDetectionTrainer(DetectionTrainer):
    """
    Custom trainer that uses VisDroneDataset for merged class training.
    """
    
    def build_dataset(self, img_path, mode="train", batch=None):
        """Build custom VisDroneDataset."""
        return VisDroneMergedDataset(
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
            self.model.names = VisDroneMergedDataset.merged_names
            self.model.nc = len(VisDroneMergedDataset.merged_names)
            
            # Also update data dictionary
            if hasattr(self, 'data'):
                self.data['names'] = VisDroneMergedDataset.merged_names
                self.data['nc'] = len(VisDroneMergedDataset.merged_names)