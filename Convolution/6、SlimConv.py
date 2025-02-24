import torch
import torch.nn as nn


class SlimConv(nn.Module):

    def __init__(self, in_planes, stride=1, groups=2, dilation=1):
        super(SlimConv, self).__init__()
        self.stride = stride

        reduce_1 = 2
        reduce_2 = 4

        self.conv2_2 = nn.Sequential(nn.Conv2d(in_planes // reduce_1, in_planes // reduce_2, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(in_planes // reduce_2),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_planes // reduce_2, in_planes // reduce_2, kernel_size=3,
                                               stride=stride, groups=groups, padding=dilation, bias=False,
                                               dilation=dilation),
                                     nn.BatchNorm2d(in_planes // reduce_2))

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_planes // reduce_1, in_planes // reduce_1, kernel_size=3, stride=stride, groups=groups,
                      padding=dilation, bias=False, dilation=dilation),
            nn.BatchNorm2d(in_planes // reduce_1))

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 32, kernel_size=1, bias=False),
                                nn.BatchNorm2d(in_planes // 32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_planes // 32, in_planes, kernel_size=1),
                                nn.Sigmoid())
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = x
        b, c, h, _ = out.size()

        w = self.pool(out)
        w = self.fc(w)
        w_f = torch.flip(w, [1])

        out1 = w * out
        out2 = w_f * out
        fs1 = torch.split(out1, c // 2, 1)
        fs2 = torch.split(out2, c // 2, 1)

        ft1 = fs1[0] + fs1[1]
        ft2 = fs2[0] + fs2[1]

        out2_1 = self.conv2_1(ft1)
        out2_2 = self.conv2_2(ft2)

        out = torch.cat((out2_1, out2_2), 1)
        return out


if __name__ == '__main__':
    """
    output_channel = (input_channel / reduce_1) + (input_channel / reduce_2)
    可以在后面接一个普通卷积调整一下
    """
    x = torch.randn(4, 512, 7, 7).cuda()
    model = SlimConv(512).cuda()
    out = model(x)
    print(out.shape)
