from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr
from eyeinthesky.dataset import VisDroneMergedDataset

class MergedClassDetectionValidator(DetectionValidator):
    """
    Custom validator that uses VisDroneDataset for validation/testing with merged classes.
    """
    
    def build_dataset(self, img_path, mode="val", batch=None):
        """Build custom VisDroneDataset for validation."""
        return VisDroneMergedDataset(
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
            self.model.names = VisDroneMergedDataset.merged_names
            if hasattr(self.data, 'names'):
                self.data['names'] = VisDroneMergedDataset.merged_names
                self.data['nc'] = len(VisDroneMergedDataset.merged_names)