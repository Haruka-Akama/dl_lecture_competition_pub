import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "/data1/akamaharuka/data/") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        print(f"Loading {split}_X.pt...")
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        print(f"{split}_X.pt loaded successfully.")

        print(f"Loading {split}_subject_idxs.pt...")
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        print(f"{split}_subject_idxs.pt loaded successfully.")
        
        if split in ["train", "val"]:
            print(f"Loading {split}_y.pt...")
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
            print(f"{split}_y.pt loaded successfully.")
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        subject_idx = self.subject_idxs[i]
        if self.y is not None:
            y = self.y[i]
            return X, y, subject_idx
        else:
            return X, subject_idx

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

# Usage example

