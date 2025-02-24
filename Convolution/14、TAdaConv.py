import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple


class TAdaConv2d(nn.Module):
    """ TAdaConv2d """
    def __init__(self, in_channels, out_channels, kernel_size=[1,3,3],
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cal_dim="cin"):
        super(TAdaConv2d, self).__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1
        assert cal_dim in ["cin", "cout"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cal_dim = cal_dim

        # base weights (W_b)
        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2])
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, alpha):
        _, _, c_out, c_in, kh, kw = self.weight.size()
        b, c_in, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(1, -1, h, w)

        if self.cal_dim == "cin":
            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(2) * self.weight).reshape(-1, c_in // self.groups, kh, kw)
        elif self.cal_dim == "cout":
            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(3) * self.weight).reshape(-1, c_in // self.groups, kh, kw)

        bias = None
        if self.bias is not None:
            bias = self.bias.repeat(b, t, 1).reshape(-1)
        output = F.conv2d(
            x, weight=weight, bias=bias, stride=self.stride[1:], padding=self.padding[1:],
            dilation=self.dilation[1:], groups=self.groups * b * t)

        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0, 2, 1, 3, 4)

        return output

    def __repr__(self):
        return f"TAdaConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, " + \
               f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, cal_dim=\"{self.cal_dim}\")"


class TAdaConv3d(nn.Module):
    """ TAdaConv3d """
    def __init__(self, in_channels, out_channels, kernel_size=[1,3,3],
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cal_dim="cin"):
        super(TAdaConv3d, self).__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert stride[0] == 1
        assert dilation[0] == 1
        assert cal_dim in ["cin", "cout"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cal_dim = cal_dim

        # base weights (W_b)
        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2])
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, alpha):
        _, _, c_out, c_in, kt, kh, kw = self.weight.size()
        b, c_in, t, h, w = x.size()
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, kt // 2, kt // 2), "constant", 0).unfold(
            dimension=2, size=kt, step=1
        ).permute(0, 2, 1, 5, 3, 4).reshape(1, -1, kt, h, w)

        if self.cal_dim == "cin":
            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(2).unsqueeze(-1) * self.weight).reshape(-1,
                                                                                                     c_in // self.groups,
                                                                                                     kt, kh, kw)
        elif self.cal_dim == "cout":
            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(3).unsqueeze(-1) * self.weight).reshape(-1,
                                                                                                     c_in // self.groups,
                                                                                                     kt, kh, kw)

        bias = None
        if self.bias is not None:
            bias = self.bias.repeat(b, t, 1).reshape(-1)
        output = F.conv3d(
            x, weight=weight, bias=bias, stride=[1] + list(self.stride[1:]), padding=[0] + list(self.padding[1:]),
            dilation=[1] + list(self.dilation[1:]), groups=self.groups * b * t)

        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0, 2, 1, 3, 4)

        return output

    def __repr__(self):
        return f"TAdaConv3d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, " + \
               f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, cal_dim=\"{self.cal_dim}\")"


if __name__ == '__main__':
    x = torch.randn(2, 64, 10, 32, 32)
    y = torch.rand(2, 64, 10, 1, 1)

    model = TAdaConv2d(64, 64)
    output = model(x, y)
    print(output.shape)

    # model = TAdaConv3d(64, 64)
    # output = model(x, y)
    # print(output.shape)
