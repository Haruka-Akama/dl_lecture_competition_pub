import os
import numpy as np
import torch
from typing import Tuple
from glob import glob
<<<<<<< HEAD
from torch.utils.data import Dataset  # インポート追加
from torchvision import transforms  # インポート追加

def global_contrast_normalization(X: torch.Tensor, s: float = 1.0, λ: float = 10.0) -> torch.Tensor:
    X_mean = X.mean(dim=1, keepdim=True)
    X = X - X_mean  # 平均を引く

    contrast = torch.sqrt(λ + (X ** 2).sum(dim=1, keepdim=True))
    X = s * X / contrast  # 正規化

    return X

class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "/workspace/dl_lecture_competition_pub/data/") -> None:
=======
from torch.utils.data import Dataset
import scipy.signal as signal
from sklearn.preprocessing import StandardScaler

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

class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data", new_sample_rate: int = 256, low_cut: float = 0.2, high_cut: float = 40.0, baseline_window: Tuple[int, int] = (0, 50)) -> None:
>>>>>>> fc66237 (baseline 1st try)
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.new_sample_rate = new_sample_rate
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.baseline_window = baseline_window
        
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))
        
        # トランスフォームの定義
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # PIL画像に変換
            transforms.Resize((224, 224)),  # サイズ変更
            transforms.Grayscale(num_output_channels=3),  # 3チャネルに変換
            transforms.ToTensor()  # テンソルに変換
        ])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
<<<<<<< HEAD
        X = torch.from_numpy(np.load(X_path)).float()
        
        X = global_contrast_normalization(X)  # グローバルコントラスト正規化の適用
        
        # 必要に応じて次元を調整
        if X.dim() == 2:  # (チャネル, 幅) -> (1, 高さ, 幅)
            X = X.unsqueeze(0)
        elif X.dim() == 3:  # (チャネル, 高さ, 幅) -> (高さ, 幅, チャネル)
            X = X.permute(1, 2, 0)

        if self.transform:
            X = self.transform(X)  # トランスフォームの適用
=======
        X = np.load(X_path)
        
        # 前処理
        sample_rate = 1200  # 元のサンプリングレート（仮定）
        X = preprocess_data(X, sample_rate, self.new_sample_rate, self.low_cut, self.high_cut, self.baseline_window)
        
        # スケーリング
        scaler = StandardScaler()
        X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        X = torch.tensor(X, dtype=torch.float32)
        
        # 必要に応じて次元を調整
        if X.dim() == 2:  # (幅, 高さ) -> (1, 幅, 高さ)
            X = X.unsqueeze(0)
        elif X.dim() == 3:  # (チャネル, 幅, 高さ) -> (高さ, 幅, チャネル)
            X = X.permute(1, 2, 0)
>>>>>>> fc66237 (baseline 1st try)
        
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
