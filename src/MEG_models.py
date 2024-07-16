import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig
import hydra  # 追加
from src.MEG_datasets import ThingsMEGDataset, ImageDataset  # 修正
from src.MEG_utils import set_seed 

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

# 仮のセンサー位置と被験者IDを生成
num_sensors = 306  # 仮のセンサー数
sensor_locs = np.random.rand(num_sensors, 2)  # 2D位置

# データをテンソルに変換
@hydra.main(version_base=None, config_path="configs", config_name="MEG_config")
def main(cfg: DictConfig):
    # data_dirを取得
    data_dir = cfg.data_dir
    
    # 実際のデータをロード
    data_path = [os.path.join(data_dir, 'train_X', fname) for fname in sorted(os.listdir(os.path.join(data_dir, 'train_X'))) if fname.endswith('.npy')]
    # データをロードし、テンソルに変換
    data = torch.load(data_path)

    # データの形状を確認し、必要ならば形状を調整
    if data.dim() == 3 and data.shape[1] > data.shape[2]:
        # 形状が [バッチサイズ, センサー数, 時間サンプル] の場合、調整不要
        pass
    elif data.dim() == 3 and data.shape[1] < data.shape[2]:
        # 形状が [バッチサイズ, 時間サンプル, センサー数] の場合、 [バッチサイズ, センサー数, 時間サンプル] に変換
        data = data.permute(0, 2, 1)

    # データをテンソルに変換
    X = torch.tensor(data, dtype=torch.float32)

sensor_locs = torch.tensor(sensor_locs, dtype=torch.float32)

# 被験者IDの設定
subject_ids = np.random.randint(0, 4, size=batch_size)  # 0から3までの被験者IDをランダムに生成
subject_ids = torch.tensor(subject_ids, dtype=torch.long)

# fclipモデルを初期化
num_input_channels = X.shape[1]
num_output_channels = 128  # 任意の出力チャネル数
num_subjects = 4  # 仮の被験者数
num_classes = 1854  # クラス数（画像カテゴリ数）
model = fclip(num_input_channels, num_output_channels, num_subjects, num_classes)

# モデルの出力を取得
Z = model(X, subject_ids, sensor_locs)
print(f"Output shape: {Z.shape}")
