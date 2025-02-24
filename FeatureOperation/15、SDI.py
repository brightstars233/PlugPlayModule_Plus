import torch
import torch.nn as nn
import torch.nn.functional as F


class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)])

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                      mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)

        return ans


if __name__ == '__main__':
    x_1 = torch.randn(4, 512, 7, 7).cuda()
    x_2 = torch.randn(4, 512, 14, 14).cuda()
    x_3 = torch.randn(4, 512, 28, 28).cuda()
    x_4 = torch.randn(4, 512, 56, 56).cuda()
    model_1 = SDI(512).cuda()
    model_2 = SDI(512).cuda()
    model_3 = SDI(512).cuda()
    model_4 = SDI(512).cuda()
    out_1 = model_1([x_1, x_2, x_3, x_4], x_1)
    out_2 = model_2([x_1, x_2, x_3, x_4], x_2)
    out_3 = model_3([x_1, x_2, x_3, x_4], x_3)
    out_4 = model_4([x_1, x_2, x_3, x_4], x_4)
    print(out_1.shape)
    print(out_2.shape)
    print(out_3.shape)
    print(out_4.shape)
