import torch
import torch.nn as nn
import torch.nn.functional as F


class PolarizedAttention(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PolarizedAttention, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)
        batch, channel, height, width = input_x.size()
        input_x = input_x.view(batch, channel, height * width)
        context_mask = self.conv_q_right(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax_right(context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))
        context = context.unsqueeze(-1)
        context = self.conv_up(context)
        mask_ch = self.sigmoid(context)
        out = x * mask_ch
        return out

    def channel_pool(self, x):
        g_x = self.conv_q_left(x)
        batch, channel, height, width = g_x.size()
        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = avg_x.size()
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
        context = torch.matmul(avg_x, theta_x)
        context = self.softmax_left(context)
        context = context.view(batch, 1, height, width)
        mask_sp = self.sigmoid(context)
        out = x * mask_sp
        return out

    def forward(self, x):
        # 并联
        # context_channel = self.spatial_pool(x)
        # context_spatial = self.channel_pool(x)
        # out = context_spatial + context_channel

        # 串联
        out = self.spatial_pool(x)
        out = self.channel_pool(out)

        return out


if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7).cuda()
    model = PolarizedAttention(512, 512).cuda()
    out = model(x)
    print(out.shape)
