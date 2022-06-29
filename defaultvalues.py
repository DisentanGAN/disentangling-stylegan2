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
    16: 256,
    32: 256,
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
    "seed": 42,
    "batch_size": 32,
    "dataloader_workers": 2,
    "classifier": "None",
    "classifier_classes": 10,
    "classifier_depth": 3,
    "checkpoint_path": 'checkpoints/',
    #"checkpoint_path": '/netscratch/checkpoints/',
    "save_checkpoint_every": 4,
}

# help strings for argparser
default_args_help = {
    "r1": "weight of the r1 regularization",
    "path_regularize": "weight of the path length regularization",
    "path_batch_shrink": "batch size reducing factor for the path length regularization (reduce memory consumption)",
    "d_reg_every": "interval of applying r1 regularization",
    "g_reg_every": "interval of applying path length regularization",
    "mixing": "probability of latent code mixing",
    "augment": "apply non leaking augmentation",
    "augment_p": "probability of applying augmentation. 0 = use adaptive augmentation",
    "ada_target": "target augmentation probability for adaptive augmentation",
    "ada_length": "target duration to reach augmentation probability for adaptive augmentation",
    "ada_every": "probability update interval of the adaptive augmentation",
    "latent": "style space latent dimensionality",
    "image_size": "img size to resize to",
    "n_mlp": "number of layer in the mapping network",
    "store_images_every": "store images per n epochs",
    "seed": "seed to reproduce experiments",
    "batch_size": "batch sizes for each gpus",
    "dataloader_workers": "number of workers for data loaders",
    "classifier": "classifier to be used",
    "classifier_classes": "number of classes for the classifier",
    "classifier_depth": "depth of classifier",
    "checkpoint_path": "path to the checkpoint to resume training",
    "save_checkpoint_every": "interval for saving checkpoints",
}
