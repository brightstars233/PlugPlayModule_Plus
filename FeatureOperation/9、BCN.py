import torch
import torch.nn as nn


class BatchNormm2D(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9, rescale=True):
        super(BatchNormm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum
        self.rescale = rescale
        if (self.rescale == True):
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        self.register_buffer('runningmean', torch.zeros(num_channels))
        self.register_buffer('runningvar', torch.ones(num_channels))

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        assert len(x.shape) == 4

        if (self.training):
            variance = torch.var(x, dim=[0, 2, 3], unbiased=False)
            mean = torch.mean(x, dim=[0, 2, 3])
            self.runningmean = (1 - self.momentum) * mean + (self.momentum) * self.runningmean
            self.runningvar = (1 - self.momentum) * variance + (self.momentum) * self.runningvar
            out = (x - mean.view([1, self.num_channels, 1, 1])) / torch.sqrt(
                variance.view([1, self.num_channels, 1, 1]) + self.epsilon)
        else:
            m = x.shape[0] * x.shape[2] * x.shape[3]
            out = (x - self.runningmean.view([1, self.num_channels, 1, 1])) / torch.sqrt(
                (m / (m - 1)) * self.runningvar.view([1, self.num_channels, 1, 1]) + self.epsilon)
        if (self.rescale == True):
            return out


class LayerNormm2D(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5):
        super(LayerNormm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        assert list(x.shape)[1] == self.num_channels
        assert len(x.shape) == 4  # 4 because len((batchsize, numchannels, height, width)) = 4
        variance, mean = torch.var(x, dim=[1, 2, 3], unbiased=False), torch.mean(x, dim=[1, 2, 3])

        out = (x - mean.view([-1, 1, 1, 1])) / torch.sqrt(variance.view([-1, 1, 1, 1]) + self.epsilon)
        return out


class BatchChannelNorm(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9):
        super(BatchChannelNorm, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum
        self.Batchh = BatchNormm2D(self.num_channels, epsilon=self.epsilon)
        self.layeer = LayerNormm2D(self.num_channels, epsilon=self.epsilon)
        # The BCN variable to be learnt
        self.BCN_var = nn.Parameter(torch.ones(self.num_channels))
        # Gamma and Beta for rescaling
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        X = self.Batchh(x)
        Y = self.layeer(x)
        out = self.BCN_var.view([1, self.num_channels, 1, 1]) * X + (
                1 - self.BCN_var.view([1, self.num_channels, 1, 1])) * Y
        out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out


if __name__ == '__main__':
    x = torch.randn(4, 512, 7, 7).cuda()
    model = BatchChannelNorm(512).cuda()
    out = model(x)
    print(out.shape)
