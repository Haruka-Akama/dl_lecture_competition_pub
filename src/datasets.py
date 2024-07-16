import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
import scipy.signal as signal
from sklearn.preprocessing import StandardScaler
from glob import glob

def preprocess_data(data, sample_rate, new_sample_rate, low_cut, high_cut, baseline_window):
    # リサンプリング
    num_samples = int(len(data) * float(new_sample_rate) / sample_rate)
    data = signal.resample(data, num_samples)
    
    # フィルタリング
    nyquist = 0.9 * new_sample_rate
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = signal.butter(1, [low, high], btype="band")
    data = signal.filtfilt(b, a, data)
    
    # ベースライン補正
    baseline = np.mean(data[baseline_window[0]:baseline_window[1]], axis=0)
    data = data - baseline
    
    return data

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "/workspace/dl_lecture_competition_pub/data/", new_sample_rate: int = 256, low_cut: float = 0.2, high_cut: float = 40.0, baseline_window: Tuple[int, int] = (0, 50)) -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

        # Load data
        print(f"Loading {split}_X data...")
        self.X = [np.load(os.path.join(data_dir, f"{split}_X", str(i).zfill(5) + ".npy")) for i in range(self.num_samples)]
        print(f"{split}_X data loaded successfully.")

        print(f"Loading {split}_subject_idxs data...")
        self.subject_idxs = [np.load(os.path.join(data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")).item() for i in range(self.num_samples)]
        print(f"{split}_subject_idxs data loaded successfully.")
        
        if split in ["train", "val"]:
            print(f"Loading {split}_y data...")
            self.y = [np.load(os.path.join(data_dir, f"{split}_y", str(i).zfill(5) + ".npy")).item() for i in range(self.num_samples)]
            # デバッグメッセージを追加して `self.y` の内容を確認
            print(f"Loaded {split}_y data: {self.y[:10]}")  # 最初の10個のサンプルを表示
            print(f"y data type: {type(self.y)}, y element type: {type(self.y[0])}, y shape: {np.array(self.y).shape}")
            assert len(torch.unique(torch.tensor(self.y))) == self.num_classes, "Number of classes do not match."
            print(f"{split}_y data loaded successfully.")
        
        # 前処理
        sample_rate = 1200  # 元のサンプリングレート（仮定）
        self.X = np.array([preprocess_data(x, sample_rate, new_sample_rate, low_cut, high_cut, baseline_window) for x in self.X])
        
        # スケーリング
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X.reshape(-1, self.X.shape[-1])).reshape(self.X.shape)

        # デバッグメッセージを追加して `self.subject_idxs` の内容を確認
        print(f"Loaded {split}_subject_idxs data: {self.subject_idxs[:10]}")  # 最初の10個のサンプルを表示
        print(f"subject_idxs data type: {type(self.subject_idxs)}, subject_idxs element type: {type(self.subject_idxs[0])}, subject_idxs shape: {np.array(self.subject_idxs).shape}")

        # Tensorに変換
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.subject_idxs = torch.tensor(self.subject_idxs, dtype=torch.long)
        if hasattr(self, 'y'):
            self.y = torch.tensor(self.y, dtype=torch.long)

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
