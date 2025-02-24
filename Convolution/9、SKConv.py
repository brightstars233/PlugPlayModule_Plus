import torch
import torch.nn as nn
from collections import OrderedDict


class SKConv(nn.Module):
    def __init__(self, features, kernels=[1,3,5,7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(features // reduction, L)
        self.convs = nn.ModuleList([])
        for i in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(features, features, kernel_size=i, padding=i // 2, groups=group)),
                    ('bn',nn.BatchNorm2d(features)),
                    ('relu',nn.ReLU(inplace=True))
                ]))
            )
        self.fc = nn.Linear(features, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(
                nn.Linear(self.d, features)
            )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        batch_size, c, _, _ = x.size()
        conv_outs = []
        # 1.split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)
        # 2.fuse
        feats_U = sum(conv_outs)
        feats_S = feats_U.mean(-1).mean(-1)
        feats_Z = self.fc(feats_S)
        # 3.select
        weights = []
        for fc in self.fcs:
            weight = fc(feats_Z)
            weights.append(weight.view(batch_size, c, 1, 1))
        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)
        # 4.fuse
        out=(attention_weights*feats).sum(0)
        return out


if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7).cuda()
    model = SKConv(512).cuda()
    out = model(x)
    print(out.shape)
