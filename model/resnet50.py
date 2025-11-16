import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

# ResNet50 Encoder (returns x2, x3, x4)
class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Encoder, self).__init__()
        resnet50 = models.resnet50(pretrained=pretrained)
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1  # Output channels 256
        self.layer2 = resnet50.layer2  # Output channels 512
        self.layer3 = resnet50.layer3  # Output channels 1024
        self.layer4 = resnet50.layer4  # Output channels 2048

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x1 = self.layer1(x)   # [B, 256, H/4, W/4]
        x2 = self.layer2(x1)  # [B, 512, H/8, W/8]
        x3 = self.layer3(x2)  # [B, 1024, H/16, W/16]
        x4 = self.layer4(x3)  # [B, 2048, H/32, W/32]
        
        return x2, x3, x4