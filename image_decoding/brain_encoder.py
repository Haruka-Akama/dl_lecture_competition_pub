import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from functools import partial
from typing import Optional, Union, Callable, List, Tuple
from termcolor import cprint

class SpatialAttention(nn.Module):
    def __init__(self, loc: torch.Tensor, D1: int, K: int, d_drop: float, flat: bool = True):
        super().__init__()

        self.flat = flat

        x, y = loc.T
        if self.flat:
            self.z_re = nn.Parameter(torch.Tensor(D1, K, K))
            self.z_im = nn.Parameter(torch.Tensor(D1, K, K))
            nn.init.kaiming_uniform_(self.z_re, a=np.sqrt(5))
            nn.init.kaiming_uniform_(self.z_im, a=np.sqrt(5))

            k_arange = torch.arange(K)
            rad1 = torch.einsum("k,c->kc", k_arange, x)
            rad2 = torch.einsum("l,c->lc", k_arange, y)
            rad = rad1.unsqueeze(1) + rad2.unsqueeze(0)
            self.register_buffer("cos", torch.cos(2 * torch.pi * rad))
            self.register_buffer("sin", torch.sin(2 * torch.pi * rad))
        else:
            self.z = nn.Parameter(torch.rand(size=(D1, K**2), dtype=torch.cfloat))

            a = []
            for k in range(K):
                for l in range(K):
                    a.append((k, l))
            a = torch.tensor(a)
            k, l = a[:, 0], a[:, 1]
            phi = 2 * torch.pi * (torch.einsum("k,x->kx", k, x) + torch.einsum("l,y->ly", l, y))
            self.register_buffer("cos", torch.cos(phi))
            self.register_buffer("sin", torch.sin(phi))

        self.spatial_dropout = SpatialDropout(loc, d_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.spatial_dropout(X)

        if self.flat:
            real = torch.einsum("dkl,klc->dc", self.z_re, self.cos)
            imag = torch.einsum("dkl,klc->dc", self.z_im, self.sin)
        else:
            real = torch.einsum("jm, me -> je", self.z.real, self.cos)
            imag = torch.einsum("jm, me -> je", self.z.imag, self.sin)

        a = F.softmax(real + imag, dim=-1)

        return torch.einsum("oi,bit->bot", a, X)

class SpatialDropout(nn.Module):
    def __init__(self, loc, d_drop):
        super().__init__()
        self.loc = loc
        self.d_drop = d_drop
        self.num_channels = loc.shape[0]

    def forward(self, X):
        assert X.shape[1] == self.num_channels

        if self.training:
            drop_center = self.loc[np.random.randint(self.num_channels)]
            distances = (self.loc - drop_center).norm(dim=-1)
            mask = torch.where(distances < self.d_drop, 0.0, 1.0).to(device=X.device)
            X = torch.einsum("c,bct->bct", mask, X)

        return X

class SubjectBlock(nn.Module):
    def __init__(self, num_subjects: int, loc: np.ndarray, D1: int, K: int, d_drop: float, num_channels: int, spatial_attention: bool = True):
        super().__init__()

        self.num_subjects = num_subjects

        if spatial_attention:
            self.spatial_attention = SpatialAttention(loc, D1, K, d_drop)
        else:
            cprint("Not using spatial attention.", "yellow")
            self.spatial_attention = None

        self.conv = nn.Conv1d(in_channels=D1 if spatial_attention else num_channels, out_channels=D1, kernel_size=1, stride=1)
        self.subject_layer = nn.ModuleList(
            [nn.Conv1d(in_channels=D1, out_channels=D1, kernel_size=1, stride=1, bias=False) for _ in range(self.num_subjects)]
        )

    def forward(self, X: torch.Tensor, subject_idxs: Optional[torch.Tensor]) -> torch.Tensor:
        if self.spatial_attention is not None:
            X = self.spatial_attention(X)

        X = self.conv(X)

        if subject_idxs is not None:
            X = torch.cat(
                [self.subject_layer[i](x.unsqueeze(dim=0)) for i, x in zip(subject_idxs, X)]
            )
        else:
            cprint("Unknown subject.", "yellow")

            X = torch.stack(
                [self.subject_layer[i](X) for i in range(self.num_subjects)]
            ).mean(dim=0)

        return X

class ConvBlock(nn.Module):
    def __init__(self, k: int, D1: int, D2: int, ksize: int = 3, p_drop: float = 0.1) -> None:
        super().__init__()

        self.k = k
        in_channels = D1 if k == 0 else D2

        self.conv0 = nn.Conv1d(in_channels=in_channels, out_channels=D2, kernel_size=ksize, padding="same", dilation=2 ** ((2 * self.k) % 5))
        self.batchnorm0 = nn.BatchNorm1d(num_features=D2)
        self.conv1 = nn.Conv1d(in_channels=D2, out_channels=D2, kernel_size=ksize, padding="same", dilation=2 ** ((2 * self.k + 1) % 5))
        self.batchnorm1 = nn.BatchNorm1d(num_features=D2)
        self.conv2 = nn.Conv1d(in_channels=D2, out_channels=2 * D2, kernel_size=ksize, padding="same", dilation=2)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.k == 0:
            X = self.conv0(X)
        else:
            X = self.conv0(X) + X

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X
        X = F.gelu(self.batchnorm1(X))

        X = self.conv2(X)
        X = F.glu(X, dim=-2)

        return self.dropout(X)

class TemporalAggregation(nn.Module):
    def __init__(self, temporal_dim: int, embed_dim: int, temporal_agg: str = "affine", multiplier: int = 1) -> None:
        super().__init__()

        if temporal_agg == "affine":
            self.layers = nn.Linear(temporal_dim, multiplier)
        elif temporal_agg == "pool":
            self.layers = nn.AdaptiveAvgPool1d(1)
        else:
            raise NotImplementedError()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)

class BrainEncoder(nn.Module):
    def __init__(self, args, subjects: Union[int, List[str]]) -> None:
        super().__init__()

        D1, D2, D3, K, F = args.D1, args.D2, args.D3, args.K, args.F
        temporal_dim = int(args.seq_len * args.brain_sfreq)
        num_clip_tokens = args.num_clip_tokens
        num_subjects: int = subjects if isinstance(subjects, int) else len(subjects)
        num_channels: int = args.num_channels
        spatial_attention: bool = args.spatial_attention
        num_blocks: int = args.num_blocks
        conv_block_ksize: int = args.conv_block_ksize
        temporal_agg: str = args.temporal_agg
        p_drop: float = args.p_drop
        d_drop: float = args.d_drop
        final_ksize: int = args.final_ksize
        final_stride: int = args.final_stride

        self.ignore_subjects = args.ignore_subjects

        loc = self._ch_locations_2d(args.montage_path)

        num_subjects = num_subjects if not self.ignore_subjects else 1
        self.subject_block = SubjectBlock(num_subjects, loc, D1, K, d_drop, num_channels, spatial_attention)

        self.blocks = nn.Sequential()

        for k in range(num_blocks):
            self.blocks.add_module(f"block{k}", ConvBlock(k, D1, D2, conv_block_ksize, p_drop))

        self.conv_final = nn.Conv1d(in_channels=D2, out_channels=D3, kernel_size=final_ksize, stride=final_stride)

        self.temporal_aggregation = TemporalAggregation(temporal_dim, D3, temporal_agg, multiplier=num_clip_tokens)

        self.clip_head = nn.Sequential(nn.LayerNorm([D3, num_clip_tokens]), nn.GELU(), nn.Conv1d(Dデータセットの修正に伴って、分別機（classifier）の修正が必要です。以下の修正点に基づいて、必要な変更を反映しました。
