import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50, resnext50_32x4d


class WeightNorm2d(nn.Module):
    def __init__(self, num_features):
        super(WeightNorm2d, self).__init__()

    def forward(self, x):
        return x


class Model(nn.Module):
    def __init__(self, backbone_type, num_classes, bn_type):
        super(Model, self).__init__()
        backbones = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnext50': resnext50_32x4d}
        bns = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'wn': WeightNorm2d}
        self.backbone = backbones[backbone_type](num_classes=num_classes, norm_layer=bns[bn_type])

    def forward(self, x):
        return self.backbone(x)
