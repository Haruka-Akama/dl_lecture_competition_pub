import os
import numpy as np
import torch
from glob import glob

t


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

rain_set = ThingsMEGDataset(split='train', data_dir='/content/drive/.shortcut-targets-by-id/1qxAFAKovUXm28TMQUFxKljs01WcTHCeY/ColabData/dl_lecture_competition_pub/data(1)')
print(f"Number of samples in train_set: {len(train_set)}")



data_dir = '/content/drive/.shortcut-targets-by-id/1qxAFAKovUXm28TMQUFxKljs01WcTHCeY/ColabData/dl_lecture_competition_pub/data(1)'
train_files = glob(os.path.join(data_dir, "train_X", "*.pt"))
print(f"Number of train files: {len(train_files)}")

loader_args = {
    'batch_size': 64,
    'num_workers': 2,
    # 他の引数があれば追加
}

train_set = ThingsMEGDataset(split='train', data_dir='/content/drive/.shortcut-targets-by-id/1qxAFAKovUXm28TMQUFxKljs01WcTHCeY/ColabData/dl_lecture_competition_pub/data(1)')
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)

if len(train_files) == 0:
    raise ValueError("No training data found. Please check the data directory and files.")