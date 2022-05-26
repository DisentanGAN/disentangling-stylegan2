import torch.nn as nn

from util import *


class MappingNetwork(nn.Module):
    def __init__(
        self,
        style_dim,
        n_mlp,
        lr_mlp=0.01, # TODO: learning rate should probably not be set here
    ):
        super().__init__()
        self.style_dim = style_dim
        layers = [PixelNorm()]

        for _ in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim,
                    style_dim,
                    lr_mul=lr_mlp,
                    activation="fused_lrelu",
                )
            )

        self.style = nn.Sequential(*layers)

    def forward(self, z):
        y = self.style(z)
        return y

