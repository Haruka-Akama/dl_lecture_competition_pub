import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.5,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        
        self.layernorm0 = nn.LayerNorm(out_dim)
        self.layernorm1 = nn.LayerNorm(out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))
        X = self.layernorm0(X.permute(0, 2, 1)).permute(0, 2, 1)

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))
        X = self.layernorm1(X.permute(0, 2, 1)).permute(0, 2, 1)

        return self.dropout(X)

class ConvLSTMClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        num_layers: int = 2,
        p_drop: float = 0.5,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            ConvBlock(in_channels, hid_dim, kernel_size=3, p_drop=p_drop),
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
        # Apply convolutional layers
        X = self.conv(X)
        # LSTM expects input of shape (batch, seq_len, input_size)
        X = X.permute(0, 2, 1)
        X, _ = self.lstm(X)
        X = X.permute(0, 2, 1)

        return self.head(X)

# 使用例
model = ConvLSTMClassifier(
    num_classes=10,
    seq_len=100,
    in_channels=64,
    hid_dim=128,
    num_layers=2,
    p_drop=0.5
)