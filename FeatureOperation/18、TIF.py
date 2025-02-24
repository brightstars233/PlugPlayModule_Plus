import torch
from torch import nn, einsum
from einops.einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TIF(nn.Module):
    def __init__(self, dim_s, dim_l):
        super().__init__()
        self.transformer_s = Transformer(dim=dim_s, depth=1, heads=3, dim_head=32, mlp_dim=128)
        self.transformer_l = Transformer(dim=dim_l, depth=1, heads=1, dim_head=64, mlp_dim=256)
        self.norm_s = nn.LayerNorm(dim_s)
        self.norm_l = nn.LayerNorm(dim_l)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear_s = nn.Linear(dim_s, dim_l)
        self.linear_l = nn.Linear(dim_l, dim_s)

    def forward(self, e, r):
       b_e, c_e, h_e, w_e = e.shape
       e = e.reshape(b_e, c_e, -1).permute(0, 2, 1)
       b_r, c_r, h_r, w_r = r.shape
       r = r.reshape(b_r, c_r, -1).permute(0, 2, 1)
       e_t = torch.flatten(self.avgpool(self.norm_l(e).transpose(1,2)), 1)
       r_t = torch.flatten(self.avgpool(self.norm_s(r).transpose(1,2)), 1)
       e_t = self.linear_l(e_t).unsqueeze(1)
       r_t = self.linear_s(r_t).unsqueeze(1)
       r = self.transformer_s(torch.cat([e_t, r],dim=1))[:, 1:, :]
       e = self.transformer_l(torch.cat([r_t, e],dim=1))[:, 1:, :]
       e = e.permute(0, 2, 1).reshape(b_e, c_e, h_e, w_e)
       r = r.permute(0, 2, 1).reshape(b_r, c_r, h_r, w_r)
       return e + r


if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7).cuda()
    y = torch.randn(4, 512, 7, 7).cuda()
    model = TIF(512, 512).cuda()
    out = model(x,y)
    print(out.shape)
