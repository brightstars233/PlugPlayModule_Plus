import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PATM(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mode='fc'):
        super().__init__()

        self.fc_h = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_c = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)

        self.tfc_h = nn.Conv2d(2 * dim, dim, (1, 7), stride=1, padding=(0, 7 // 2), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d(2 * dim, dim, (7, 1), stride=1, padding=(7 // 2, 0), groups=dim, bias=False)
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        # 对h和w都学出相位
        if mode == 'fc':
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
        else:
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
                                              nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
                                              nn.BatchNorm2d(dim), nn.ReLU())

    def forward(self, x):
        B, C, H, W = x.shape
        # C, H, W = x.shape
        # 相位
        theta_h = self.theta_h_conv(x)
        theta_w = self.theta_w_conv(x)
        # Channel-FC提取振幅
        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        # 用欧拉公式对特征进行展开
        x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)
        # Token-FC
        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WaveBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PATM(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop, mode=mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7).cuda()
    model = WaveBlock(512).cuda()
    out = model(x)
    print(out.shape)
