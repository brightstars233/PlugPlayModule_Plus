import torch
import torch.nn as nn


class SPABlock(nn.Module):
    def __init__(self, in_channels, k=8, adaptive = False, reduction=16, learning=False, mode='pow'):
        super(SPABlock, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.k = k
        self.adptive = adaptive
        self.reduction = reduction
        self.learing = learning
        if self.learing is True:
            self.k = nn.Parameter(torch.tensor(self.k))

        self.mode = mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x, return_info=False):
        input_shape = x.shape
        if len(input_shape)==4:
            x = x.view(x.size(0), self.in_channels, -1)
            x = x.permute(0, 2, 1)
        batch_size,N = x.size(0),x.size(1)

        #（B, H*W，C）
        if self.mode == 'pow':
            x_pow = torch.pow(x,2)# （batchsize，H*W，channel）
            x_powsum = torch.sum(x_pow,dim=2)# （batchsize，H*W）

        if self.adptive is True:
            self.k = N//self.reduction
            if self.k == 0:
                self.k = 1

        outvalue, outindices = x_powsum.topk(k=self.k, dim=-1, largest=True, sorted=True)

        outindices = outindices.unsqueeze(2).expand(batch_size, self.k, x.size(2))
        out = x.gather(dim=1, index=outindices).to(self.device)

        if return_info is True:
            return out, outindices, outvalue
        else:
            return out


if __name__ == '__main__':
    """
    输入：[B, C, H, W] / [B, H*W, C]
    输出：[B, k, C]
    """
    x = torch.randn(4, 512, 7, 7)
    model = SPABlock(in_channels=512, k=7*7)
    output = model(x)
    print(output.shape)
