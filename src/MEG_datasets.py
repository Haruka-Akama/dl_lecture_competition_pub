import os
import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "/data1/akamaharuka/data-omni/") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = 65728
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
        X = torch.from_numpy(np.load(X_path)).float()
        
        # 必要に応じて次元を調整
        if X.dim() == 2:  # (チャネル, 幅) -> (1, 高さ, 幅)
            X = X.unsqueeze(0)
        elif X.dim() == 3:  # (チャネル, 高さ, 幅) -> (高さ, 幅, チャネル)
            X = X.permute(1, 2, 0)

        if self.transform:
            X = self.transform(X)  # トランスフォームの適用
        
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
        sample = np.load(self.X_paths[0])
        return sample.shape[0]

    @property
    def seq_len(self) -> int:
        sample = np.load(self.X_paths[0])
        return sample.shape[1]

class ImageDataset(Dataset):
    def __init__(self, split: str, images_dir: str = "/data1/akamaharuka/Images", data_dir: str = "/data1/akamaharuka/data-omni", transform=None):
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.images_dir = images_dir
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 画像パスとラベルを取得
        self.image_paths = glob(os.path.join(images_dir, '**', '*.jpg'), recursive=True)
        self.relative_paths = [os.path.relpath(path, images_dir) for path in self.image_paths]
        self.labels = [os.path.dirname(rel_path).replace(os.sep, '/') for rel_path in self.relative_paths]

    def __getitem__(self, i):
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path))
        
        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))
            return y, subject_idx
        else:
            return y, subject_idx

    
    def __len__(self) -> int:
        return len(self.image_paths)    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        if self.y is not None:
            label = self.y[idx]
            return image, label, self.image_files[idx]
        else:
            return image, self.image_files[idx]
        




    @property
    def num_subjects(self) -> int:
        return len(np.unique(self.subject_idxs))

    @property
    def num_classes(self) -> int:
        return len(np.unique(self.y)) if self.y is not None else None
