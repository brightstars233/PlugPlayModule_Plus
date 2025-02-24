import torch
import torch.nn as nn


class ECAModule(nn.Module):
    """
    ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    papers: https://arxiv.org/pdf/1910.03151v4.pdf
    inputs: tensor:(4, 512, 7, 7)
    outputs: tensor:(4, 512, 7, 7)
    """
    def __init__(self, channel, k_size=3):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7).cuda()
    model = ECAModule(512).cuda()
    out = model(x)
    print(out.shape)