import os
import numpy as np
import torch
from glob import glob

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data(1)") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.pt")))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".pt")
        X = torch.load(X_path)  # 修正

        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".pt")
        subject_idx = torch.load(subject_idx_path)  # 修正

        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".pt")
            y = torch.load(y_path)  # 修正
            
            return X, y, subject_idx
        else:
            return X, subject_idx
        
    @property
    def num_channels(self) -> int:
        return torch.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.pt")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return torch.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.pt")).shape[1]
