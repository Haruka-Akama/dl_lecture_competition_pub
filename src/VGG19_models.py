import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG19Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super(VGG19Classifier, self).__init__()
        self.vgg19 = models.vgg19_bn(pretrained=True)
        
        # 最後の全結合層を変更して、クラス数に適応
        self.vgg19.classifier[6] = nn.Linear(4096, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg19(x)

# 使用例
model = VGG19Classifier(num_classes=1854)
