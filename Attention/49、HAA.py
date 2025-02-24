import torch
import torch.nn as nn


def expend_as(tensor, rep):
    return tensor.repeat(1, rep, 1, 1)


class Channelblock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(Channelblock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        conv1 = self.conv1(x)

        conv2 = self.conv2(x)

        combined = torch.cat([conv1, conv2], dim=1)
        pooled = self.global_avg_pool(combined)
        pooled = torch.flatten(pooled, 1)
        sigm = self.fc(pooled)

        a = sigm.view(-1, sigm.size(1), 1, 1)
        a1 = 1 - sigm
        a1 = a1.view(-1, a1.size(1), 1, 1)

        y = conv1 * a
        y1 = conv2 * a1

        combined = torch.cat([y, y1], dim=1)
        out = self.conv3(combined)

        return out


class Spatialblock(nn.Module):
    def __init__(self, in_channels, out_channels, size):

        super(Spatialblock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=size, padding=(size // 2)),
            nn.BatchNorm2d(out_channels)
        )


    def forward(self, x, channel_data):
        conv1 = self.conv1(x)

        spatil_data = self.conv2(conv1)

        data3 = torch.add(channel_data, spatil_data)
        data3 = torch.relu(data3)
        data3 = nn.Conv2d(data3.size(1), 1, kernel_size=1, padding=0).cuda()(data3)
        data3 = torch.sigmoid(data3)

        a = expend_as(data3, channel_data.size(1))
        y = a *channel_data

        a1 = 1 - data3
        a1 = expend_as(a1, spatil_data.size(1))
        y1 = a1 * spatil_data

        combined = torch.cat([y, y1], dim=1)
        out = self.final_conv(combined)

        return out


class HAAM(nn.Module):
    def __init__(self, in_channels, out_channels, size=3):

        super(HAAM, self).__init__()

        self.channel_block = Channelblock(in_channels, out_channels)
        self.spatial_block = Spatialblock(out_channels, out_channels, size)

    def forward(self, x):
        channel_data = self.channel_block(x)

        haam_data = self.spatial_block(x, channel_data)
        return haam_data

if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7).cuda()
    model = HAAM(512, 512).cuda()
    output = model(x)
    print(output.shape)
