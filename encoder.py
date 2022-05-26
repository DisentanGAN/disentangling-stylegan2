import torch.nn as nn

from util import *

"""
Encoder gets a real image X
or a fake image x as input
and outputs an embedding
y which should be learned to be
close to a latent vector w
that would generate x when
fed to the Generator
"""

class Encoder(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        blur_kernel=[1, 3, 3, 1],
        channels=channels,
    ):
        super().__init__()

        convs = [ConvLayer(3, channels[size], 1)]
        log_size = int(math.log(size, 2))
        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)
        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(
                channels[4] * 4 * 4, channels[4], activation="fused_lrelu"
            ),
            EqualLinear(channels[4], style_dim),
        )

    def forward(self, x):
        """
        x: image
        returns: embedding of x in w-mannifold
        """
        out = self.convs(x)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group,
            -1,
            self.stddev_feat,
            channel // self.stddev_feat,
            height,
            width,
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

