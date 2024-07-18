import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from omegaconf import DictConfig 

class BasicLSTMClassifier(nn.Module):
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
            dropout=p_drop if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(p_drop)
        self.head = nn.Sequential(
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        # (b, c, t) -> (b, t, c)
        X = X.permute(0, 2, 1)
        X, _ = self.lstm(X)
        X = self.dropout(X[:, -1, :])  # 末尾の時間ステップの出力を使用
        return self.head(X)

# 他のコードと互換性を保つための設定
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")