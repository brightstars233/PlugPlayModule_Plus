import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from einops.einops import rearrange


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class GraphLayer(nn.Module):
    def __init__(self, dim=64, k1=40, k2=20):
        super(GraphLayer, self).__init__()
        self.dim = dim
        self.k1 = k1
        self.k2 = k2

        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim)

        self.conv1 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn3 = nn.BatchNorm2d(self.dim)
        self.bn4 = nn.BatchNorm2d(self.dim)
        self.bn5 = nn.BatchNorm2d(self.dim)

        self.conv3 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                    self.bn3,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
                                    self.bn4,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x_knn1 = get_graph_feature(x, k=self.k1)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x_knn1 = self.conv1(x_knn1)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x_knn1 = self.conv2(x_knn1)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x_k1 = x_knn1.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x_knn2 = get_graph_feature(x, self.k2)
        x_knn2 = self.conv3(x_knn2)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x_knn2 = self.conv4(x_knn2)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x_k1 = x_k1.unsqueeze(-1).repeat(1, 1, 1, self.k2)

        out = torch.cat([x_knn2, x_k1], dim=1)

        out = self.conv5(out)
        out = out.max(dim=-1, keepdim=False)[0]

        return out


if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7).cuda()
    """
    input : (B, C, H*W,)
    output : (B, C, H*W)
    """
    x = rearrange(x, 'b c h w -> b (h w) c')
    x = x.permute(0, 2, 1)

    model = GraphLayer(512).cuda()
    out = model(x)

    out = out.permute(0, 2, 1)
    out = rearrange(out, 'b (h w) c -> b c h w', h=7, w=7)
    print(out.shape)
