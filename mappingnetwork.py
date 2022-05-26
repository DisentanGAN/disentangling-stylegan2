import torch.nn as nn

from util import *


"""
Transforms randomly generated vector z
into some latent representation w, that
during training is supposed to achieve
some kind of disentanglement as observed
in the original StyleGAN2 paper.

We are aiming to further enforce the
disentanglement by adding downstream
tasks to the training scheme.
"""

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

    def module_name(self):
        return self.__str__().split("(")[0]

    def forward(self, z):
        """
        z: randomly sampled vector
        returns: transformation of z into w-mannifold
        """
        w = self.style(z)
        return w

