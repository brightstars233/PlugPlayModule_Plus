import torch
from torch import nn, einsum
import torch.nn.functional as F


class PGM(nn.Module):
    def __init__(self, prompt_dim, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PGM, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B, 1,
                                                                                                                  1, 1,
                                                                                                                  1,
                                                                                                                  1).squeeze(
            1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt


if __name__ == '__main__':
    x = torch.randn(4, 3, 64, 64).cuda()
    model = PGM(3, lin_dim=3).cuda()
    out = model(x)
    print(out.shape)
