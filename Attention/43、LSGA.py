import torch
import torch.nn as nn
from einops import rearrange, repeat


class LSGAttention(nn.Module):
    def __init__(self, dim, att_inputsize, num_heads=4, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.att_inputsize = att_inputsize[0]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        totalpixel = self.att_inputsize * self.att_inputsize
        gauss_coords_h = torch.arange(totalpixel) - int((totalpixel - 1) / 2)
        gauss_coords_w = torch.arange(totalpixel) - int((totalpixel - 1) / 2)
        gauss_x, gauss_y = torch.meshgrid([gauss_coords_h, gauss_coords_w])
        sigma = 10
        gauss_pos_index = torch.exp(torch.true_divide(-(gauss_x ** 2 + gauss_y ** 2), (2 * sigma ** 2)))
        self.register_buffer("gauss_pos_index", gauss_pos_index)
        self.token_wA = nn.Parameter(torch.empty(1, self.att_inputsize * self.att_inputsize, dim),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, dim, dim),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        wa = repeat(self.token_wA, '() n d -> b n d', b=B_)  # wa (bs 4 64)
        wa = rearrange(wa, 'b h w -> b w h')  # Transpose # wa (bs 64 4)
        A = torch.einsum('bij,bjk->bik', x, wa)  # A (bs 81 4)
        A = rearrange(A, 'b h w -> b w h')  # Transpose # A (bs 4 81)
        A = A.softmax(dim=-1)
        VV = repeat(self.token_wV, '() n d -> b n d', b=B_)  # VV(bs,64,64)
        VV = torch.einsum('bij,bjk->bik', x, VV)  # VV(bs,81,64)
        x = torch.einsum('bij,bjk->bik', A, VV)  # T(bs,4,64)
        absolute_pos_bias = self.gauss_pos_index.unsqueeze(0)
        q = self.qkv(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = x.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + absolute_pos_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    H, W = 7, 7
    x = torch.randn(4, 512, 7, 7)
    x = rearrange(x, 'b c h w -> b (h w) c')
    model = LSGAttention(512, (7, 7))
    output = model(x)
    output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
    print(output.shape)
