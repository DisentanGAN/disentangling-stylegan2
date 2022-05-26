import torch.nn as nn

from util import *


"""
Input is y, the embedding generated
by the encoder based upon an input
image x.

Output is true/false -- whether the
embedding y is based upon a 
synthesized image x or a real image X
"""

class Discriminator(nn.Module):
    def __init__(self, style_dim):
        super().__init__()

        self.final_linear = nn.Sequential(
            EqualLinear(style_dim, style_dim, activation="fused_lrelu"),
            EqualLinear(style_dim, style_dim, activation="fused_lrelu"),
            EqualLinear(style_dim, style_dim, activation="fused_lrelu"),
            EqualLinear(style_dim, 1),
        )

    def forward(self, y):
        """
        y: embedding of image, generated by encoder(x)
        returns: prediction whether y is embedding of
        a real image X or a fake image x.
        """
        return self.final_linear(y)

