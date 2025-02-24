import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward


class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)

        return x


if __name__ == '__main__':
    """
    input : (B, C, H, W)
    output : (B, C, H / 2, W / 2)
    """
    x = torch.randn(4, 512, 7, 7).cuda()
    model = HWD(512, 512).cuda()
    out = model(x)
    print(out.shape)