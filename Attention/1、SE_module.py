import torch
import torch.nn as nn


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Networks
    papers: https://arxiv.org/pdf/1709.01507v4.pdf
    inputs: tensor:(4, 512, 7, 7)
    outputs: tensor:(4, 512, 7, 7)
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7).cuda()
    model = SELayer(512).cuda()
    out = model(x)
    print(out.shape)