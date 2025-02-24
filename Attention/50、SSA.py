import numpy as np
import torch
from torch import nn
from torch.nn import init
from einops.einops import rearrange


class SimplifiedScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, h=8, dropout=.1):
        super(SimplifiedScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model//h
        self.d_v = d_model//h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


if __name__ == '__main__':
    H, W = 7, 7
    x = torch.randn(4, 512, 7, 7)
    x = rearrange(x, 'b c h w -> b (h w) c')
    model = SimplifiedScaledDotProductAttention(512)
    output = model(x,x,x)
    output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
    print(output.shape)
