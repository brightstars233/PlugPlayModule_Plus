import torch
import torch.nn as nn


class PSAModule(nn.Module):
    """
    EPSANet: An Efficient Pyramid Squeeze Attention Block on Convolutional Neural Network
    papers: https://arxiv.org/pdf/2105.14447v2.pdf
    inputs: tensor:(4, 512, 7, 7)
    outputs: tensor:(4, 512, 7, 7)
    """
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = nn.Conv2d(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                                stride=stride, groups=conv_groups[0])
        self.conv_2 = nn.Conv2d(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                                stride=stride, groups=conv_groups[1])
        self.conv_3 = nn.Conv2d(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                                stride=stride, groups=conv_groups[2])
        self.conv_4 = nn.Conv2d(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                                stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7).cuda()
    model = PSAModule(512, 512).cuda()
    out = model(x)
    print(out.shape)