import torch
import torch.nn as nn
import torchvision.models as models

class VGG19Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super(VGG19Classifier, self).__init__()
        self.vgg19 = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
        
        # 最初の畳み込み層を再定義
        self.vgg19.features[0] = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        
        # 最後の全結合層を変更して、クラス数に適応
        self.vgg19.classifier[6] = nn.Linear(4096, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg19(x)
