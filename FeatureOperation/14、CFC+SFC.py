import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    # (1, 3, 6, 8)
    # (1, 4, 8,12)
    def __init__(self, grids=(1, 3, 6, 8), channels=256):
        super(PSPModule, self).__init__()

        self.grids = grids
        self.channels = channels

    def forward(self, feats):
        b, c, h, w = feats.size()

        ar = w / h

        return torch.cat([
            F.adaptive_avg_pool2d(feats, (self.grids[0], max(1, round(ar * self.grids[0])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[1], max(1, round(ar * self.grids[1])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[2], max(1, round(ar * self.grids[2])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[3], max(1, round(ar * self.grids[3])))).view(b, self.channels, -1)
        ], dim=2)


class LocalAttenModule(nn.Module):
    def __init__(self, in_channels=256, inter_channels=64):
        super(LocalAttenModule, self).__init__()

        self.conv = nn.Sequential(
            ConvModule(in_channels, inter_channels, 1, 1, 0),
            nn.Conv2d(inter_channels, in_channels, kernel_size=3, padding=1, bias=False))

        self.tanh_spatial = nn.Tanh()
        self.conv[1].weight.data.zero_()

    def forward(self, x):
        res1 = x
        res2 = x

        x = self.conv(x)
        x_mask = self.tanh_spatial(x)

        res1 = res1 * x_mask

        return res1 + res2


class CFC(nn.Module):
    def __init__(self, in_channels=512, inter_channels=256, grids=(6, 3, 2, 1)):  # 先ce后ffm
        super(CFC, self).__init__()
        self.grids = grids
        self.inter_channels = inter_channels

        self.reduce_channel = ConvModule(in_channels, inter_channels, 3, 1, 1)

        self.query_conv = nn.Conv2d(in_channels=inter_channels, out_channels=32, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=inter_channels, out_channels=32, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=inter_channels, out_channels=self.inter_channels, kernel_size=1)
        self.key_channels = 32

        self.value_psp = PSPModule(grids, inter_channels)
        self.key_psp = PSPModule(grids, inter_channels)

        self.softmax = nn.Softmax(dim=-1)

        self.local_attention = LocalAttenModule(inter_channels, inter_channels // 4)

    def forward(self, x):
        x = self.reduce_channel(x)  # 降维- 128

        m_batchsize, _, h, w = x.size()

        query = self.query_conv(x).view(m_batchsize, 32, -1).permute(0, 2, 1)  ##  b c n ->  b n c

        key = self.key_conv(self.key_psp(x))  ## b c s

        sim_map = torch.matmul(query, key)

        sim_map = self.softmax(sim_map)
        # sim_map = self.attn_drop(sim_map)
        value = self.value_conv(self.value_psp(x))  # .permute(0,2,1)  ## b c s

        # context = torch.matmul(sim_map,value) ## B N S * B S C ->  B N C
        context = torch.bmm(value, sim_map.permute(0, 2, 1))  # B C S * B S N - >  B C N

        # context = context.permute(0,2,1).view(m_batchsize,self.inter_channels,h,w)
        context = context.view(m_batchsize, self.inter_channels, h, w)
        # out = x + self.gamma * context
        context = self.local_attention(context)
        out = x + context

        return out


class SFC(nn.Module):
    def __init__(self, in_channel=128):
        super(SFC, self).__init__()

        self.conv_8 = ConvModule(in_channel, in_channel, 3, 1, 1)
        self.cp1x1 = nn.Conv2d(in_channel, in_channel // 4, 1, bias=False)

        self.conv_32 = ConvModule(in_channel, in_channel, 3, 1, 1)
        self.sp1x1 = nn.Conv2d(in_channel, in_channel // 4, 1, bias=False)

        self.groups = 2

        self.conv_offset = nn.Sequential(
            ConvModule(in_channel // 2, in_channel // 2, 1, 1, 0),
            nn.Conv2d(in_channel // 2, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False))

        self.conv_offset[1].weight.data.zero_()

    def forward(self, cp, sp):
        n, _, out_h, out_w = cp.size()

        # 深层特征
        sp = self.conv_32(sp)  # 语义特征  1 / 8  256
        sp = F.interpolate(sp, cp.size()[2:], mode='bilinear', align_corners=True)
        # 浅层特征
        cp = self.conv_8(cp)

        ## 将cp1x1/sp1x1和conv_offset 合并，将导致更低的计算参数
        cp1x1 = self.cp1x1(cp)
        sp1x1 = self.sp1x1(sp)

        conv_results = self.conv_offset(torch.cat([cp1x1, sp1x1], 1))

        sp = sp.reshape(n * self.groups, -1, out_h, out_w)
        cp = cp.reshape(n * self.groups, -1, out_h, out_w)

        offset_l = conv_results[:, 0:self.groups * 2, :, :].reshape(n * self.groups, -1, out_h, out_w)
        offset_h = conv_results[:, self.groups * 2:self.groups * 4, :, :].reshape(n * self.groups, -1, out_h, out_w)

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(sp).to(sp.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n * self.groups, 1, 1, 1).type_as(sp).to(sp.device)

        grid_l = grid + offset_l.permute(0, 2, 3, 1) / norm
        grid_h = grid + offset_h.permute(0, 2, 3, 1) / norm

        cp = F.grid_sample(cp, grid_l, align_corners=True)  ## 考虑是否指定align_corners
        sp = F.grid_sample(sp, grid_h, align_corners=True)  ## 考虑是否指定align_corners

        cp = cp.reshape(n, -1, out_h, out_w)
        sp = sp.reshape(n, -1, out_h, out_w)

        att = 1 + torch.tanh(conv_results[:, self.groups * 4:, :, :])
        sp = sp * att[:, 0:1, :, :] + cp * att[:, 1:2, :, :]

        return sp


if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7).cuda()
    y = torch.randn(4, 256, 14, 14).cuda()
    cfc = CFC(512, 256).cuda()
    sfc = SFC(256).cuda()
    out = cfc(x)
    out = sfc(y, out)
    print(out.shape)
