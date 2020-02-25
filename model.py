import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnext50_32x4d


class WeightNorm2d(nn.Module):
    def __init__(self, num_features):
        super(WeightNorm2d, self).__init__()

    def forward(self, x):
        return x


class Model(nn.Module):
    def __init__(self, data_name, backbone_type, num_classes, norm_type):
        super(Model, self).__init__()
        backbones = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnext50': resnext50_32x4d}
        norms = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'wn': WeightNorm2d}
        backbone = backbones[backbone_type](num_classes=num_classes, norm_layer=norms[norm_type])
        self.feature = []
        for name, module in backbone.named_children():
            if data_name != 'imagenet':
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if isinstance(module, nn.MaxPool2d):
                    continue
            if isinstance(module, nn.Linear):
                self.fc = module
            else:
                self.feature.append(module)
        self.feature = nn.Sequential(*self.feature)

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
