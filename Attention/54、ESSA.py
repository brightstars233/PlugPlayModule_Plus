import torch
import torch.nn as nn
import math
from einops.einops import rearrange


class ESSAttn(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.lnqkv = nn.Linear(dim, dim * 3)
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
        b, N, C = x.shape
        qkv = self.lnqkv(x)
        qkv = torch.split(qkv, C, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.mean(q, dim=2, keepdim=True)
        q = q - a
        a = torch.mean(k, dim=2, keepdim=True)
        k = k - a
        q2 = torch.pow(q, 2)
        q2s = torch.sum(q2, dim=2, keepdim=True)
        k2 = torch.pow(k, 2)
        k2s = torch.sum(k2, dim=2, keepdim=True)
        t1 = v
        k2 = torch.nn.functional.normalize((k2 / (k2s + 1e-7)), dim=-2)
        q2 = torch.nn.functional.normalize((q2 / (q2s + 1e-7)), dim=-1)
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)
        attn = t1 + t2
        attn = self.ln(attn)
        return attn

    def is_same_matrix(self, m1, m2):
        rows, cols = len(m1), len(m1[0])
        for i in range(rows):
            for j in range(cols):
                if m1[i][j] != m2[i][j]:
                    return False
        return True


if __name__ == '__main__':
    H, W = 7, 7
    x = torch.randn(4, 512, 7, 7)
    x = rearrange(x, 'b c h w -> b (h w) c')
    model = ESSAttn(512)
    output = model(x)
    output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
    print(output.shape)
