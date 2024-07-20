import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig
import hydra  # 追加
from src.MEG_datasets import ThingsMEGDataset, ImageDataset  # 修正
from src.MEG_utils import set_seed 
from torchvision import transforms

class SpatialAttentionLayer(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, harmonics=32):
        super(SpatialAttentionLayer, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.harmonics = harmonics
        self.fourier_weights = nn.Parameter(torch.randn(num_output_channels, harmonics, harmonics, 2))

    def forward(self, X, sensor_locs):
        sensor_locs = (sensor_locs - sensor_locs.min(0)[0]) / (sensor_locs.max(0)[0] - sensor_locs.min(0)[0])
        attention_maps = []
        for j in range(self.num_output_channels):
            a_j = torch.zeros(sensor_locs.shape[0])
            for k in range(self.harmonics):
                for l in range(self.harmonics):
                    real_part = self.fourier_weights[j, k, l, 0]
                    imag_part = self.fourier_weights[j, k, l, 1]
                    a_j += real_part * torch.cos(2 * np.pi * (k * sensor_locs[:, 0] + l * sensor_locs[:, 1])) + \
                           imag_part * torch.sin(2 * np.pi * (k * sensor_locs[:, 0] + l * sensor_locs[:, 1]))
            attention_maps.append(a_j)
        attention_maps = torch.stack(attention_maps, dim=1)
        attention_weights = F.softmax(attention_maps, dim=-1)
        return torch.einsum('bi,bjk->bjk', attention_weights, X)

class SubjectSpecificLayer(nn.Module):
    def __init__(self, num_channels, num_subjects):
        super(SubjectSpecificLayer, self).__init__()
        self.num_channels = num_channels
        self.num_subjects = num_subjects
        self.subject_matrices = nn.Parameter(torch.randn(num_subjects, num_channels, num_channels))

    def forward(self, X, subject_id):
        M_s = self.subject_matrices[subject_id]
        return torch.einsum('bij,bjk->bik', X, M_s)

class ResidualDilatedConvBlock(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(ResidualDilatedConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(num_input_channels, num_output_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(num_output_channels, num_output_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(num_output_channels, 2 * num_output_channels, kernel_size=3, padding=4, dilation=4)
        self.batch_norm = nn.BatchNorm1d(num_output_channels)
        self.gelu = nn.GELU()
        self.glu = nn.GLU(dim=1)

    def forward(self, X):
        X = self.conv1(X)
        X = self.batch_norm(X)
        X = self.gelu(X)
        X = self.conv2(X)
        X = self.batch_norm(X)
        X = self.gelu(X)
        X = self.conv3(X)
        X = self.batch_norm(X)
        X = self.glu(X)
        return X

class fclip(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, num_subjects, num_classes):
        super(fclip, self).__init__()
        self.spatial_attention = SpatialAttentionLayer(num_input_channels, 270)
        self.subject_specific = SubjectSpecificLayer(270, num_subjects)
        self.residual_blocks = nn.ModuleList([ResidualDilatedConvBlock(270, 320) for _ in range(5)])
        self.conv_final1 = nn.Conv1d(320, 640, kernel_size=1)
        self.gelu = nn.GELU()
        self.conv_final2 = nn.Conv1d(640, num_output_channels, kernel_size=1)
        self.classifier = nn.Linear(num_output_channels, num_classes)

    def forward(self, X, subject_ids, sensor_locs):
        X = self.spatial_attention(X, sensor_locs)
        X = self.subject_specific(X, subject_ids)
        for block in self.residual_blocks:
            X = block(X)
        X = self.conv_final1(X)
        X = self.gelu(X)
        X = self.conv_final2(X)
        X = torch.mean(X, dim=-1)  # グローバル平均プーリング
        X = self.classifier(X)
        return X

