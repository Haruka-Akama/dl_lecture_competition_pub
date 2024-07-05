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
            self.y = torch.load(os.path.join(data_path, 