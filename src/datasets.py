import os
import numpy as np
import torch
from PIL import Image

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", transform=None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.transform = transform

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

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        if X.dim() == 3:  # チャネル、高さ、幅
            X = X.permute(1, 2, 0)  # 高さ、幅、チャネルに変換
        if self.transform:
            X = self.transform(Image.fromarray(X.numpy()))
        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]