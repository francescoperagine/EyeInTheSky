from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr, LOGGER
from eyeinthesky.dataset import VisDroneDataset
from ultralytics.nn.tasks import DetectionModel

class MergedClassDetectionTrainer(DetectionTrainer):
    """
    Custom YOLO trainer that uses the VisDroneDataset with merged classes.
    
    Extends the standard DetectionTrainer to work with the merged-class dataset.
    The key modifications are in build_dataset() to use VisDroneDataset instead of
    the default, and in set_model_attributes() to properly update the model's class
    names and count to match the merged dataset.
    
    This ensures that all aspects of training - from data loading to loss calculation -
    work consistently with the merged class structure.
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
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Create and return a DetectionModel."""
        
        model = DetectionModel(
            cfg=cfg, 
            nc=self.data["nc"],
            verbose=verbose,
        )

        model.args = self.args
        
        if weights:
            LOGGER.info(f"Loading weights into model")
            model.load(weights)
            
        return model    
    
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