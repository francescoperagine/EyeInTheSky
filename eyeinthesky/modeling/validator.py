from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr
from eyeinthesky.dataset import VisDroneDataset

class MergedClassDetectionValidator(DetectionValidator):
    """
    Custom validator for evaluating models trained on merged VisDrone classes.
    
    Works in tandem with MergedClassDetectionTrainer to ensure that validation
    uses the same class merging as training. The build_dataset() method creates
    VisDroneDataset instances for validation, and set_model_attributes() updates
    the model's class configuration to match the merged dataset.
    
    This allows for consistent metrics calculation across training and evaluation.
    """
    
    def build_dataset(self, img_path, mode="val", batch=None):
        """Build custom VisDroneDataset for validation."""
        return VisDroneDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch or self.args.batch,
            augment=False,
            hyp=self.args,
            rect=self.args.rect,
            cache=None,
            single_cls=self.args.single_cls,
            stride=self.stride,
            pad=0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=None,
            data=self.data,
        )
       
    def set_model_attributes(self):
        """Update model attributes for merged classes if using a PyTorch model."""
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