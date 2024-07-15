import os, sys
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import gc
from termcolor import cprint
from tqdm import tqdm
from typing import Tuple, List

class ThingsMEGCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()

        self.preproc_dir = os.path.join(args.save_dir, "preproc")

        self.large_test_set = args.large_test_set
        self.num_clip_tokens = args.num_clip_tokens
        self.align_token = args.align_token

        # Load preprocessed data
        self.train_X = torch.load(os.path.join(self.preproc_dir, "train_X.pt"))
        self.val_X = torch.load(os.path.join(self.preproc_dir, "val_X.pt"))
        self.test_X = torch.load(os.path.join(self.preproc_dir, "test_X.pt"))

        self.train_subject_idxs = torch.load(os.path.join(self.preproc_dir, "train_subject_idxs.pt"))
        self.val_subject_idxs = torch.load(os.path.join(self.preproc_dir, "val_subject_idxs.pt"))
        self.test_subject_idxs = torch.load(os.path.join(self.preproc_dir, "test_subject_idxs.pt"))

        self.train_y = torch.load(os.path.join(self.preproc_dir, "train_y.pt"))
        self.val_y = torch.load(os.path.join(self.preproc_dir, "val_y.pt"))

        self.train_Y = torch.load(os.path.join(self.preproc_dir, "train_Y.pt"))
        self.val_Y = torch.load(os.path.join(self.preproc_dir, "val_Y.pt"))

        # Combine train and validation sets
        self.X = torch.cat([self.train_X, self.val_X], dim=0)
        self.subject_idxs = torch.cat([self.train_subject_idxs, self.val_subject_idxs], dim=0)
        self.y = torch.cat([self.train_y, self.val_y], dim=0)
        self.Y = torch.cat([self.train_Y, self.val_Y], dim=0)

        if args.chance:
            self.X = self.X[torch.randperm(len(self.X))]

        del self.train_X, self.val_X, self.train_subject_idxs, self.val_subject_idxs, self.train_y, self.val_y, self.train_Y, self.val_Y
        gc.collect()

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.subject_idxs[i], self.y[i]  # Adjusted to return y instead of categories and high_categories

    def _extract_token(self, Y: torch.Tensor) -> torch.Tensor:
        if Y.ndim == 2:
            assert self.num_clip_tokens == 1, "num_clip_tokens > 1 is specified, but the embeddings don't have temporal dimension."
            assert not self.align_token == "all", "align_token is specified as 'all', but the embeddings don't have temporal dimension."

            Y = Y.unsqueeze(1)
        else:
            if self.align_token == "mean":
                assert self.num_clip_tokens == 1
                Y = Y.mean(dim=1, keepdim=True)
            elif self.align_token == "cls":
                assert self.num_clip_tokens == 1
                Y = Y[:, :1]
            else:
                assert self.align_token == "all"

        return Y

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

    @property
    def brain_sfreq(self) -> float:
        return self._sampling_frequency
