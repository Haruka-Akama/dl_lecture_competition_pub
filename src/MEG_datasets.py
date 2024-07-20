import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "/data1/akamaharuka/data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        print(f"Loading {split}_X.pt...")
        self.X_paths = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        print(f"{split}_X.pt loaded successfully.")

        print(f"Loading {split}_subject_idxs.pt...")
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        print(f"{split}_subject_idxs.pt loaded successfully.")
        
        if split in ["train", "val"]:
            print(f"Loading {split}_y.pt...")
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
            print(f"{split}_y.pt loaded successfully.")
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.X_paths)

    def __getitem__(self, i):
        data_dir = "/data1/akamaharuka/data"
        X = np.load(os.path.join(data_dir, f"train_X.pt"))
        subject_idx = self.subject_idxs[i]
        
        if self.y is not None:
            y = self.y[i]
            return X, y, subject_idx
        else:
            return X, subject_idx

    @property
    def num_channels(self) -> int:
        data_dir = "/data1/akamaharuka/data"
        sample = np.load(os.path.join(data_dir, f"train_X.pt"[0]))
        return sample.shape[0]

    @property
    def seq_len(self) -> int:
        data_dir = "/data1/akamaharuka/data"
        sample = np.load(data_dir, f"train_X.pt"[0])
        return sample.shape[1]

class ImageDataset(Dataset):
    def __init__(self, split: str, images_dir: str = "/data1/akamaharuka/Images", data_dir: str = "/data1/akamaharuka/data/", transform=None):
        self.images_dir = images_dir
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        print(f"Loading Images directory '{images_dir}'...")
        self.image_paths = []

        # ディレクトリを再帰的に探索し、すべての画像ファイルパスをリストに追加
        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.endswith('.jpg'):
                    # ファイルパスをリストに追加
                    self.image_paths.append(os.path.join(root, file))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image files found in directory '{images_dir}'")

        print(f"Found {len(self.image_paths)} images.")

        print(f"Loading {split}_subject_idxs.pt...")
        subject_dir = os.path.join(data_dir, f"{split}_subject_idxs.pt")
        try:
            self.subject_idxs = torch.load(subject_dir)
            print(f"{split}_subject_idxs.pt loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load {subject_dir}: {e}")

        if split in ["train", "val"]:
            print(f"Loading {split}_y.pt...")
            label_dir = os.path.join(data_dir, f"{split}_y.pt")
            try:
                self.y = torch.load(label_dir)
                print(f"{split}_y.pt loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load {label_dir}: {e}")
        else:
            self.y = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        if self.y is not None:
            label = self.y[idx]
            return image, label
        else:
            return image


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        if self.y is not None:
            label = self.y[idx]
            return image, label
        else:
            return image 

    @property
    def num_subjects(self) -> int:
        return len(np.unique(self.subject_idxs))


    @property
    def num_classes(self) -> int:
        return len(np.unique(self.y)) if self.y is not None else None

# Usage example:
# from torchvision import transforms
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
# train_set = ThingsMEGDataset(split='train', data_dir='path/to/data')
# image_set = ImageDataset(split='train', images_dir='path/to/Images', data_dir='path/to/data', transform=transform)
