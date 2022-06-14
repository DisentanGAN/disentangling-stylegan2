"""
Here we define default parameters
for the models.
Should also be controllable by
command line args.
"""

import torch


channel_multiplier = 2
channels = {
    4: 32,
    8: 32,
    16: 16,
    32: 16,
    # 64: 256 * channel_multiplier,
    # 128: 128 * channel_multiplier,
    # 256: 64 * channel_multiplier,
    # 512: 32 * channel_multiplier,
    # 1024: 16 * channel_multiplier,
}

# values here only for debbuging
# TODO: find sensible default values
optim_conf = {
    "generator": {
        "optimizer": torch.optim.Adam,
        "args": {"lr" : 0.002},
    },
    "discriminator": {
        "optimizer": torch.optim.Adam,
        "args": {"lr" : 0.002},
    },
    "encoder": {
        "optimizer": torch.optim.Adam,
        "args": {"lr": 0.002},
    },
    "mapping": {
        "optimizer": torch.optim.Adam,
        "args": {"lr": 0.002}
    },
    "classifier": {
        "optimizer": torch.optim.Adam,
        "args": {"lr": 0.01}
    },
}

default_args = {
    "r1": 10,
    "path_regularize": 2, 
    "path_batch_shrink": 2, 
    "d_reg_every": 16,
    "g_reg_every": 4,
    "mixing": 0.9,
    "augment": False,
    "augment_p": 0.8,
    "ada_target": 0.6,
    "ada_length": 500000,
    "ada_every": 256,
    "latent": 128,
    "image_size": 32,
    "n_mlp": 8,
    "store_images_every": 1,
    "num_example_images": 8,
    "seed": 42,
    "batch_size": 32,
    "dataloader_workers": 2,
    "classifier": "Linear",
    "classifier_classes": 10,
}
