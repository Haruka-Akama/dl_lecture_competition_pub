import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
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

        # データ増強の定義
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # テンソルをPIL画像に変換
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),  # PIL画像をテンソルに変換
        ])

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        if X.dim() == 2:  # チャネル次元がない場合は追加
            X = X.unsqueeze(0)

        # データ増強を適用
        if self.transform:
            X = self.transform(X)

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

# Example usage
# dataset = ThingsMEGDataset("train", "data_path")
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
