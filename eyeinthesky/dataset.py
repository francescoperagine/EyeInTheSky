from typing import Dict
from torch.utils.data import Dataset

class MergedDataset(Dataset):
    def __init__(self, original_dataset, merge_mapping: Dict[int, int]):
        """
        Wraps an existing YOLO dataset to remap labels on the fly.
        
        Args:
            original_dataset: The dataset returned by build_yolo_dataset.
            merge_mapping: Dict mapping original class indices to a unified class index.
                           For example: {1: 0, 2: 0} to merge classes 1 and 2 into class 0 ("person").
        """
        self.original_dataset = original_dataset
        self.merge_mapping = merge_mapping

    def __getitem__(self, idx):
        # Get the original item; expected format: (img, labels, ...)
        item = self.original_dataset[idx]
        # Assume labels are stored in the second element as a numpy array of shape (N, 5)
        # with the first column being the class index.
        if len(item) >= 2:
            img, labels = item[0], item[1]
            if labels is not None and len(labels) > 0:
                new_labels = labels.copy()
                for i in range(len(new_labels)):
                    orig_cls = int(new_labels[i, 0])
                    if orig_cls in self.merge_mapping:
                        new_labels[i, 0] = self.merge_mapping[orig_cls]
                item = (img, new_labels) + tuple(item[2:])
        return item

    def __len__(self):
        return len(self.original_dataset)