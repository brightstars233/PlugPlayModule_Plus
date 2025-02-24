import torch
import torch.nn as nn


class CSRA(nn.Module):  # one basic block
    """
    这个注意力 的输出 维度是 [B, C]，所以需要[B, C, H, W]的还需要进一步处理一下
    """
    def __init__(self, input_dim, num_classes, T, lam):
        super(CSRA, self).__init__()
        self.T = T  # temperature
        self.lam = lam  # Lambda
        self.head = nn.Conv2d(input_dim, num_classes, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x (B d H W)
        # normalize classifier
        # score (B C HxW)
        score = self.head(x) / torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0, 1)
        score = score.flatten(2)
        base_logit = torch.mean(score, dim=2)

        if self.T == 99:  # max-pooling
            att_logit = torch.max(score, dim=2)[0]
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = torch.sum(score * score_soft, dim=2)

        return base_logit + self.lam * att_logit


class MHA(nn.Module):  # multi-head attention
    temp_settings = {  # softmax temperature settings
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        6: [1, 2, 3, 4, 5, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(self, input_dim, num_classes, num_heads=1, lam=0.1):
        super(MHA, self).__init__()
        self.temp_list = self.temp_settings[num_heads]
        self.multi_head = nn.ModuleList([
            CSRA(input_dim, num_classes, self.temp_list[i], lam)
            for i in range(num_heads)
        ])

    def forward(self, x):
        logit = 0.
        for head in self.multi_head:
            logit += head(x)
        return logit


if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7).cuda()
    model = MHA(512, 2).cuda()
    out = model(x)
    print(out.shape)
