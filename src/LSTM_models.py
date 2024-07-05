import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class ConvLSTMClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hid_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.lstm = nn.LSTM(
            input_size=hid_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b t h -> b (t h)"),
            nn.Linear(hid_dim * 2, num_classes),  # * 2 for bidirectional
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        # Apply