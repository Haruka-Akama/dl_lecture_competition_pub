import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from glob import glob


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "/data1/akamaharuka/data-omni/") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

        # Load data
        print(f"Loading {split}_X data...")
        self.X = [np.load(os.path.join(data_dir, f"{split}_X", str(i).zfill(5) + ".npy")) for i in range(self.num_samples)]
        self.subject_idxs = [np.load(os.path.join(data_dir, f"{split}_subject_idxs", str(i).zfill(5) + ".npy")) for i in range(self.num_samples)]
        if self.split in ["train", "val"]:
            self.y = [np.load(os.path.join(data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")) for i in range(self.num_samples)]
        print(f"{split}_X data loaded successfully.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = torch.from_numpy(np.load(X_path))
        
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path))
        
        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))
            
            return X, y, subject_idx
        else:
            return X, subject_idx

    @property
    def num_channels(self) -> int:
        return self.X[0].shape[1]

    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]
