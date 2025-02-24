import torch
import torch.nn as nn
from torch.nn import init


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class FA(nn.Module):
    def __init__(self, channel):
        super(FA, self).__init__()
        self.calayer = CALayer(channel)
        self.palayer = PALayer(channel)

    def forward(self, x):
        x = self.calayer(x)
        res = self.palayer(x)
        return res


if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7)
    model = FA(512)
    output = model(x)
    print(output.shape)
