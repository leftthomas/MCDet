from torch import nn
from torch.nn import functional as F


class FastResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.head = Head(in_channels)
        self.body = Body()
        self.tail = Tail()
        self.classifier = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(batch_size, -1)
        x = self.classifier(x)
        return x


class Head(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv = ConvBlock(in_channels=in_channels, out_channels=32, stride=2)
        self.dsconv1 = nn.Sequential(
            # depthwise convolution
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, dilation=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            # pointwise convolution
            nn.Conv2d(32, 48, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.dsconv2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, dilation=1, groups=48, bias=False),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 64, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class Body(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(Bottleneck(64, 64, 2, 6),
                                    Bottleneck(64, 64, 1, 6))
        self.layer2 = nn.Sequential(Bottleneck(64, 96, 2, 6),
                                    Bottleneck(96, 96, 1, 6))
        self.layer3 = nn.Sequential(Bottleneck(96, 128, 1, 6),
                                    Bottleneck(128, 128, 1, 6))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Tail(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(in_channels=128, out_channels=192, stride=2)
        self.dsconv1 = nn.Sequential(
            # depthwise convolution
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, dilation=1, groups=192, bias=False),
            nn.BatchNorm2d(192),
            # pointwise convolution
            nn.Conv2d(192, 256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.dsconv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 320, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True))
        self.conv2 = ConvBlock(in_channels=320, out_channels=512, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv2(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, groups=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv(input)
        return self.relu(self.bn(x))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()

        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
