"""
Here we define default parameters
for the models.
Should also be controllable by
command line args.
"""

import torch


channel_multiplier = 2
channels = {
    4: 512,
    8: 512,
    16: 512,
    32: 512,
    64: 256 * channel_multiplier,
    128: 128 * channel_multiplier,
    256: 64 * channel_multiplier,
    512: 32 * channel_multiplier,
    1024: 16 * channel_multiplier,
}

# this is only a sketch =)
optimizers = {
    "default": {
        "optimizer": torch.optim.Adam,
        "args": {"lr" : 0.2},
    },
    "MappingNetwork": {
        "optimizer": torch.optim.Adam,
        "args": {"lr" : 0.2},
    },
}

