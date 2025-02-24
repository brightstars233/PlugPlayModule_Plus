import torch
import math
from torch import nn
from torch.nn import functional as F


"""
这个注意力要求输入的维度为 (B, H, W, C),
所以输入前如果为(B, C, H, W)的，可以先变(B, H, W, C)，然后注意力处理，处理完在变回(B, C, H, W)
"""


class OutlookAttention(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim**-0.5

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size**4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W

        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads,
                                   self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H

        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
            self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, C * self.kernel_size * self.kernel_size, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)

        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)

        return x


if __name__ == '__main__':
    x = torch.randn(4, 64, 128, 128).cuda()
    model = OutlookAttention(64, 8).cuda()
    # 变 (B, H, W, C)
    x = x.permute(0, 2, 3, 1)
    out = model(x)
    # 变 (B, C, H, W)
    out = out.permute(0, 3, 1, 2)
    print(out.shape)