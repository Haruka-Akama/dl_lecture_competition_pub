import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.context_vector = nn.Parameter(torch.Tensor(hidden_dim, 1))
        nn.init.xavier_uniform_(self.attention_weights)
        nn.init.xavier_uniform_(self.context_vector)

    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_len, hidden_dim)
        u = torch.tanh(torch.matmul(hidden_states, self.attention_weights))  # (batch_size, seq_len, hidden_dim)
        a = torch.matmul(u, self.context_vector).squeeze(-1)  # (batch_size, seq_len)
        attention_scores = F.softmax(a, dim=-1)  # (batch_size, seq_len)
        attention_scores = attention_scores.unsqueeze(-1)  # (batch_size, seq_len, 1)
        weighted_sum = torch.sum(hidden_states * attention_scores, dim=1)  # (batch_size, hidden_dim)
        return weighted_sum

class BasicLSTMClassifierWithAttention(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        num_layers: int = 2,
        p_drop: float = 0.5
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=p_drop if num_layers > 1 else 0,
            bidirectional=True
        )

        self.attention = Attention(hid_dim * 2)  # 双方向LSTMのため * 2
        self.dropout = nn.Dropout(p_drop)
        self.head = nn.Sequential(
            nn.Linear(hid_dim * 2, num_classes),  # 双方向のため * 2
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.permute(0, 2, 1)  # (batch_size, seq_len, in_channels)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(X)  # (batch_size, seq_len, hid_dim * 2)
        attention_out = self.attention(lstm_out)  # (batch_size, hid_dim * 2)
        attention_out = self.dropout(attention_out)
        return self.head(attention_out)
