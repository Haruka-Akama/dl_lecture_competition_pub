import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import resample, butter, filtfilt
from sklearn.preprocessing import StandardScaler

class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data", resample_freq: int = None, filter_freq: tuple = None, baseline_correction: bool = False, original_freq: int = 1000, baseline_period: int = 100) -> None:
        super().__init__()
        self.original_freq = original_freq
        self.baseline_period = baseline_period
        # 以降のコード
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
        
        # Resampling
        if resample_freq is not None:
            self.resample_data(resample_freq)
        
        # Filtering
        if filter_freq is not None:
            self.filter_data(filter_freq)
        
        # Baseline correction
        if baseline_correction:
            self.baseline_correct()
        
        # Scaling
        self.scale_data()
    
    def resample_data(self, new_freq: int):
        original_len = self.X.shape[2]
        new_len = int(original_len * new_freq / self.original_freq)
        self.X = torch.tensor(resample(self.X.numpy(), new_len, axis=2))
        print(f"Data resampled to {new_freq}Hz.")
    
    def filter_data(self, freq: tuple):
        b, a = butter(4, freq, btype='bandpass', fs=self.original_freq)
        self.X = torch.tensor(filtfilt(b, a, self.X.numpy(), axis=2))
        print(f"Data filtered with bandpass filter: {freq}Hz.")
    
    def baseline_correct(self):
        baseline = self.X[:, :, :self.baseline_period].mean(axis=2, keepdims=True)
        self.X = self.X - baseline
        print("Baseline correction applied.")
    
    def scale_data(self):
        scaler = StandardScaler()
        for i in range(self.X.shape[0]):
            self.X[i] = torch.tensor(scaler.fit_transform(self.X[i].T).T)
        print("Data scaling applied.")
    
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
