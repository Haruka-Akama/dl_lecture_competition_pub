import os
import numpy as np
import torch
from glob import glob

def global_contrast_normalization(X: torch.Tensor, s: float = 1.0, λ: float = 10.0) -> torch.Tensor:
    X_mean = X.mean(dim=1, keepdim=True)
    X = X - X_mean  # Subtract the mean

    contrast = torch.sqrt(λ + (X ** 2).sum(dim=1, keepdim=True))
    X = s * X / contrast  # Normalize

    return X

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "/workspace/dl_lecture_competition_pub/data/") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = torch.from_numpy(np.load(X_path))
        X = global_contrast_normalization(X)  # Apply Global Contrast Normalization
        
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
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]

# Example usage:
# dataset = ThingsMEGDataset(split="train", data_dir="/workspace/dl_lecture_competition_pub/data/")
