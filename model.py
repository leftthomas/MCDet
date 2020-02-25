import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50, resnext50_32x4d


class Model(nn.Module):
    def __init__(self, backbone_type, num_classes, with_bn=True):
        super(Model, self).__init__()
        backbones = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnext50': resnext50_32x4d}
        self.backbone = backbones[backbone_type](num_classes=num_classes)
        if not with_bn:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    del module

    def forward(self, x):
        return self.backbone(x)
