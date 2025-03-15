from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import colorstr, LOGGER
import numpy as np

class VisDroneMergedDataset(YOLODataset):
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
