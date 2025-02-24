import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP)
    """

    def __init__(self, channel, bias=True):
        super().__init__()
        self.w_1 = nn.Conv3d(channel, channel, bias=bias, kernel_size=1)
        self.w_2 = nn.Conv3d(channel, channel, bias=bias, kernel_size=1)

    def forward(self, x):
        return self.w_2(F.tanh(self.w_1(x)))


class PSCA(nn.Module):
    """ Progressive Spectral Channel Attention (PSCA)
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Conv3d(d_model, d_ff, 1, bias=False)
        self.w_2 = nn.Conv3d(d_ff, d_model, 1, bias=False)
        self.w_3 = nn.Conv3d(d_model, d_model, 1, bias=False)

        nn.init.zeros_(self.w_3.weight)

    def forward(self, x):
        x = self.w_3(x) * x + x
        x = self.w_1(x)
        x = F.gelu(x)
        x = self.w_2(x)
        return x


class MHRSA(nn.Module):
    """ Multi-Head Recurrent Spectral Attention
    """

    def __init__(self, channels, multi_head=True, ffn=True):
        super().__init__()
        self.channels = channels
        self.multi_head = multi_head
        self.ffn = ffn

        if ffn:
            self.ffn1 = MLP(channels)
            self.ffn2 = MLP(channels)

    def _conv_step(self, inputs):
        if self.ffn:
            Z = self.ffn1(inputs).tanh()
            F = self.ffn2(inputs).sigmoid()
        else:
            Z, F = inputs.split(split_size=self.channels, dim=1)
            Z, F = Z.tanh(), F.sigmoid()
        return Z, F

    def _rnn_step(self, z, f, h):
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, reverse=False):
        Z, F = self._conv_step(inputs)

        if self.multi_head:
            Z1, Z2 = Z.split(self.channels // 2, 1)
            Z2 = torch.flip(Z2, [2])
            Z = torch.cat([Z1, Z2], dim=1)

            F1, F2 = F.split(self.channels // 2, 1)
            F2 = torch.flip(F2, [2])
            F = torch.cat([F1, F2], dim=1)

        h = None
        h_time = []

        if not reverse:
            for _, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for _, (z, f) in enumerate((zip(
                    reversed(Z.split(1, 2)), reversed(F.split(1, 2))
            ))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)

        y = torch.cat(h_time, dim=2)

        if self.multi_head:
            y1, y2 = y.split(self.channels // 2, 1)
            y2 = torch.flip(y2, [2])
            y = torch.cat([y1, y2], dim=1)

        return y


if __name__ == '__main__':
    x = torch.randn(4, 512, 1, 16, 16).cuda()
    model = PSCA(512, 512).cuda()
    # model = MHRSA(512).cuda()
    output = model(x)
    print(output.shape)
