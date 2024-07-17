import torch
import torch.nn as nn
import torchvision.models as models

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

# VGG19 のプリトレーニングされたモデルを使用する
def get_vgg19_model(num_classes, pretrained=True):
    if pretrained:
        model = models.vgg19_bn(pretrained=True)
    else:
        model = models.vgg19_bn()
    
    # 最後の分類層を置き換える
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model
