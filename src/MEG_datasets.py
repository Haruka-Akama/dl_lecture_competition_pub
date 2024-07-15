import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "orkspace/dl_lecture_competition_pub/data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        print(f"Loading {split}_X directory...")
        self.X_paths = [os.path.join(data_dir, f"{split}_X", fname) for fname in sorted(os.listdir(os.path.join(data_dir, f"{split}_X"))) if fname.endswith('.npy')]
        print(f"{split}_X directory loaded successfully.")

        print(f"Loading {split}_subject_idxs directory...")
        subject_dir = os.path.join(data_dir, f"{split}_subject_idxs")
        self.subject_idxs = [np.load(os.path.join(subject_dir, fname)) for fname in sorted(os.listdir(subject_dir)) if fname.endswith('.npy')]
        self.subject_idxs = np.concatenate(self.subject_idxs, axis=0)
        print(f"{split}_subject_idxs loaded successfully.")
        
        if split in ["train", "val"]:
            print(f"Loading {split}_y directory...")
            label_dir = os.path.join(data_dir, f"{split}_y")
            self.y = [np.load(os.path.join(label_dir, fname)) for fname in sorted(os.listdir(label_dir)) if fname.endswith('.npy')]
            self.y = np.concatenate(self.y, axis=0)
            assert len(np.unique(self.y)) == self.num_classes, "Number of classes do not match."
            print(f"{split}_y loaded successfully.")
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.X_paths)

    def __getitem__(self, i):
        X = np.load(self.X_paths[i])
        subject_idx = self.subject_idxs[i]
        
        if self.y is not None:
            y = self.y[i]
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
    def __init__(self, split: str, images_dir: str = "workspace/dl_lecture_competition_pub/data/Images", data_dir: str = "workspace/dl_lecture_competition_pub/data", transform=None):
        self.images_dir = images_dir
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        print(f"Loading {split}_image_paths.txt...")
        with open(os.path.join(images_dir, f"split"), 'r') as file:
            self.image_paths = [line.strip() for line in file]
        print(f"{split}_image_paths.txt loaded successfully.")
        
        print(f"Loading {split}_subject_idxs directory...")
        subject_dir = os.path.join(data_dir, f"{split}_subject_idxs")
        self.subject_idxs = [np.load(os.path.join(subject_dir, fname)) for fname in sorted(os.listdir(subject_dir)) if fname.endswith('.npy')]
        self.subject_idxs = np.concatenate(self.subject_idxs, axis=0)
        print(f"{split}_subject_idxs loaded successfully.")
        
        if split in ["train", "val"]:
            print(f"Loading {split}_y directory...")
            label_dir = os.path.join(data_dir, f"{split}_y")
            self.y = [np.load(os.path.join(label_dir, fname)) for fname in sorted(os.listdir(label_dir)) if fname.endswith('.npy')]
            self.y = np.concatenate(self.y, axis=0)
            print(f"{split}_y loaded successfully.")
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_paths[idx] + '.jpg')
        image = Image.open(image_path).convert('RGB')
        subject_idx = self.subject_idxs[idx]
        
        if self.transform:
            image = self.transform(image)
        
        if self.y is not None:
            label = self.y[idx]
            return image, label, subject_idx
        else:
            return image, subject_idx

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
