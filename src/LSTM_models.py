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
<<<<<<< HEAD
        num_layers: int = 2,
        p_drop: float = 0.5,
=======
        num_blocks: int = 4,
        kernel_size: int = 5,
        num_subjects: int = 4,
        subject_emb_dim: int = 32,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 2,
        dropout_prob: float = 0.5,
        weight_decay: float = 1e-5
>>>>>>> fc66237 (baseline 1st try)
    ) -> None:
        super(ConvLSTMClassifier, self).__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.hid_dim = hid_dim
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.num_subjects = num_subjects
        self.subject_emb_dim = subject_emb_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.dropout_prob = dropout_prob
        self.weight_decay = weight_decay

<<<<<<< HEAD
        self.conv = nn.Sequential(
            ConvBlock(in_channels, hid_dim, kernel_size=3, p_drop=p_drop),
            nn.MaxPool1d(kernel_size=2)
        )
=======

        self.blocks = nn.Sequential(*[
            ConvBlock(in_channels if i == 0 else hid_dim, hid_dim, kernel_size=kernel_size, p_drop=dropout_prob)
            for i in range(num_blocks)
        ])
        
        
        self.subject_embedding = nn.Embedding(num_subjects, subject_emb_dim)
        
        self.batchnorm = nn.BatchNorm1d(hid_dim)
        self.layernorm = nn.LayerNorm(hid_dim + subject_emb_dim)
>>>>>>> fc66237 (baseline 1st try)

        self.lstm = nn.LSTM(
            input_size=hid_dim + subject_emb_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_prob
        )

        self.dropout = nn.Dropout(dropout_prob)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(lstm_hidden_dim, num_classes),
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

<<<<<<< HEAD
# 使用例
model = ConvLSTMClassifier(
    num_classes=10,
    seq_len=100,
    in_channels=64,
    hid_dim=128,
    num_layers=2,
    p_drop=0.5
)
=======
>>>>>>> fc66237 (baseline 1st try)
