import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


def conv_1x1(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class SelfAttentionBlock(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels,
                 key_query_num_convs, value_out_num_convs):
        super(SelfAttentionBlock, self).__init__()
        self.key_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
        )
        self.query_project = self.buildproject(
            in_channels=query_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs
        )
        self.value_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=value_out_num_convs
        )
        self.out_project = self.buildproject(
            in_channels=transform_channels,
            out_channels=out_channels,
            num_convs=value_out_num_convs
        )
        self.transform_channels = transform_channels

    def forward(self, query_feats, key_feats, value_feats):
        batch_size = query_feats.size(0)

        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()  # (B, h*w, C)

        key = self.key_project(key_feats)
        key = key.reshape(*key.shape[:2], -1)  # (B, C, h*w)

        value = self.value_project(value_feats)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()  # (B, h*w, C)

        sim_map = torch.matmul(query, key)

        sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)  # (B, h*w, K)

        context = torch.matmul(sim_map, value)  # (B, h*w, C)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])  # (B, C, h, w)

        context = self.out_project(context)  # (B, C, h, w)
        return context

    def buildproject(self, in_channels, out_channels, num_convs):
        convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        for _ in range(num_convs - 1):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        if len(convs) > 1:
            return nn.Sequential(*convs)
        return convs[0]


class TFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TFF, self).__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xA, xB):
        x_diff = xA - xB

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))

        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))

        xA = A_weight * xA
        xB = B_weight * xB

        x = self.catconv(torch.cat([xA, xB], dim=1))

        return x


class SFF(nn.Module):
    def __init__(self, in_channel):
        super(SFF, self).__init__()
        self.conv_small = conv_1x1(in_channel, in_channel)
        self.conv_big = conv_1x1(in_channel, in_channel)
        self.catconv = conv_3x3(in_channel * 2, in_channel)
        self.attention = SelfAttentionBlock(
            key_in_channels=in_channel,
            query_in_channels=in_channel,
            transform_channels=in_channel // 2,
            out_channels=in_channel,
            key_query_num_convs=2,
            value_out_num_convs=1
        )

    def forward(self, x_small, x_big):
        img_size = x_big.size(2), x_big.size(3)
        x_small = F.interpolate(x_small, img_size, mode="bilinear", align_corners=False)
        x = self.conv_small(x_small) + self.conv_big(x_big)
        new_x = self.attention(x, x, x_big)

        out = self.catconv(torch.cat([new_x, x_big], dim=1))
        return out


if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7).cuda()
    y = torch.randn(4, 512, 7, 7).cuda()
    # model = TFF(512, 512).cuda()
    model = SFF(512).cuda()
    out = model(x, y)
    print(out.shape)
