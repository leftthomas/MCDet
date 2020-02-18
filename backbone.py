import torch
from torch import nn
from torch.nn import functional as F


class ECAModule(nn.Module):
    """Constructs a ECA module.
    Args:
        k_size (int): the kernel size of Conv1d
    """

    def __init__(self, k_size=3):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, k_size):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = inp * expand_ratio
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])
        layers.append(ECAModule(k_size))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ECAMobileNetV2(nn.Module):
    def __init__(self, last_channel=1280, num_classes=1000):
        super(ECAMobileNetV2, self).__init__()
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = 32
        self.conv = ConvBNReLU(3, input_channel, stride=2)

        # building inverted residual blocks
        for i, (t, c, n, s) in enumerate(inverted_residual_setting):
            modules = []
            for j in range(n):
                if c < 96:
                    k_size = 1
                else:
                    k_size = 3
                stride = s if j == 0 else 1
                modules.append(InvertedResidual(input_channel, c, stride, expand_ratio=t, k_size=k_size))
                input_channel = c
            modules = nn.Sequential(*modules)
            self.add_module('block{}'.format(i + 1), modules)

        # building last layer
        self.last_conv = ConvBNReLU(input_channel, last_channel, kernel_size=1)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.last_conv(x)
        x = torch.flatten(F.adaptive_avg_pool2d(x, output_size=(1, 1)), start_dim=1)
        x = self.classifier(x)
        return x


def eca_mobilenet_v2(pretrained=False, last_channel=1280, num_classes=1000):
    """
    Constructs a ECA_MobileNetV2 architecture from pre-trained on ImageNet.
    Args:
        pretrained (bool): if True, returns a model pre-trained on ImageNet
        last_channel: (int): the out channel of last Conv2d
        num_classes: (int): the num classes of fc output
    """
    model = ECAMobileNetV2(last_channel=last_channel, num_classes=num_classes)
    if pretrained:
        state_dict = torch.load('results/backbone.pth', map_location='cpu')
        if state_dict['last_conv.0.weight'].size(0) != last_channel:
            state_dict.pop('last_conv.0.weight')
            state_dict.pop('last_conv.1.weight')
            state_dict.pop('last_conv.1.bias')
            state_dict.pop('last_conv.1.running_mean')
            state_dict.pop('last_conv.1.running_var')
            state_dict.pop('last_conv.1.num_batches_tracked')
        if state_dict['classifier.1.weight'].size() != (num_classes, last_channel):
            state_dict.pop('classifier.1.weight')
            state_dict.pop('classifier.1.bias')
        model.load_state_dict(state_dict, strict=False)
    return model
