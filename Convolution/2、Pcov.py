import torch
import torch.nn as nn


class PointWiseConv(nn.Module):
    def __init__(self, in_channels):
        super(PointWiseConv, self).__init__()
        self.pwcov = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               groups=1)

    def forward(self, x):
        return self.pwcov(x)


class Pconv(nn.Module):
    """
    Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks
    papers: https://arxiv.org/pdf/2303.03667v3.pdf
    一般配合 pwconv 或 其他卷积
    inputs: tensor:(4, 512, 7, 7)
    outputs: tensor:(4, 512, 7, 7)
    """

    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7).cuda()
    model = Pconv(512).cuda()
    # pw = PointWiseConv(512).cuda()
    out = model(x)
    # out = pw(out)
    print(out.shape)
