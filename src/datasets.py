import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint


def global_contrast_normalization(X: torch.Tensor, s: float = 1.0, λ: float = 10.0) -> torch.Tensor:
    """
    Perform Global Contrast Normalization on a single sample.
    
    Args:
        X (torch.Tensor): Input tensor with shape (channels, seq_len).
        s (float): Scale parameter.
        λ (float): Regularization parameter.
    
    Returns:
        torch.Tensor: Normalized tensor.
    """
    X_mean = X.mean(dim=1, keepdim=True)
    X = X - X_mean  # Subtract the mean

    contrast = torch.sqrt(λ + (X ** 2).sum(dim=1, keepdim=True))
    X = s * X / contrast  # Normalize

    return X

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "/kaggle/input/") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        if split == "train":
            data_path = os.path.join(data_dir, "megdata-train")
        elif split == "val":
            data_path = os.path.join(data_dir, "megdata-val")
        else:
            data_path = os.path.join(data_dir, "megdata-test")
        
        self.X = torch.load(os.path.join(data_path, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_path, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_path, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        # Apply Global Contrast Normalization
        self.X = torch.stack([global_contrast_normalization(x) for x in self.X])

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

# Example usage:
# dataset = ThingsMEGDataset(split="train", data_dir="/kaggle/input/")
