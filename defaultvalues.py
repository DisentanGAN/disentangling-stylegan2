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

# values here only for debbuging
# TODO: find sensible default values
optim_conf = {
    "default": {
        "optimizer": torch.optim.Adam,
        "args": {"lr" : 0.2},
    },
    "discrimination": {
        "optimizer": torch.optim.Adam,
        "args": {"lr" : 0.7},
    },
    "generation": {
        "optimizer": torch.optim.RMSprop,
        "args": {"lr": 0.002, "alpha": 0.19},
    },
    "consistency": {
        "optimizer": torch.optim.Adam,
        "args": {"lr": 0.612}
    },
}


args = {
    "iter": 80000,
    "batch": 32,
    "n_sample": 64,
    "r1": 10,
    "path_regularize": 2, 
    "path_batch_shrink": 2, 
    "d_reg_every": 16,
    "g_reg_every": 4,
    "mixing": 0.9,
    "ckpt": None,
    "lr": 0.002,
    "channel_multiplier": 2,
    "wandb": False,
    "local_rank": 0,
    "augment": True,
    "augment_p": 0,
    "ada_target": 0.6,
    "ada_length": 500000,
    "ada_every": 256,
    "gpu": 1,
    "name": "First test experiment",
    "run_name": "First test run",
}



