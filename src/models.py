import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 512,
        num_subjects: int = 4,
        subject_emb_dim: int = 32,
        dropout_prob: float = 0.5
    ) -> None:
        super().__init__()

        self.subject_embedding = nn.Embedding(num_subjects, subject_emb_dim)
        self.input_projection = nn.Linear(in_channels + subject_emb_dim, hid_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout_prob
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
            nn.Dropout(dropout_prob)  # Head にもドロップアウトを追加
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        subject_emb = self.subject_embedding(subject_idxs)
        subject_emb = subject_emb.unsqueeze(1).expand(-1, X.shape[1], -1)
        X = torch.cat([X, subject_emb], dim=-1)
        
        X = self.input_projection(X)
        X = self.dropout1(X.permute(1, 0, 2))  # 変換して入力にドロップアウトを追加
        X = self.transformer_encoder(X)
        X = self.dropout2(X.permute(1, 0, 2))  # Transformerの後にドロップアウトを追加

        return self.head(X.permute(0, 2, 1))

# Usage example
model = TransformerClassifier(
    num_classes=10,
    seq_len=100,
    in_channels=64,
    hid_dim=128,
    num_layers=4,
    num_heads=8,
    ff_dim=512,
    dropout_prob=0.5
)
#
