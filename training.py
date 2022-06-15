
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from disentangledsg import DisentangledSG
from defaultvalues import default_args
from datamodules import MNISTDataModule, PCAMDataModule


def train(parsed_args):
    # unite parsed and default args as dicts for DSG, override respective default values
    hparams = default_args | vars(parsed_args)

    if parsed_args.ckpt:
        dsg = DisentangledSG.load_from_checkpoint(checkpoint_path=parsed_args.ckpt)
    else:
        dsg = DisentangledSG(hparams)

    #wandb_logger = WandbLogger(project="disentangling-gan")

    datasets = {'mnist': MNISTDataModule,
                'pcam': PCAMDataModule}
    datamodule = datasets[parsed_args.dataset]()

    trainer = pl.Trainer(
        ## DEVICE ##
        gpus=parsed_args.gpu,
        #strategy='ddp',
        ## LOGGING AND CALLBACKS
        #callbacks=[],
        #logger=wandb_logger,
        ## TRAIN DURATION ##
        max_epochs=parsed_args.max_epochs,
        #max_time="00:12:00:00",
        ## DEBUG OPTIONS ##
        #fast_dev_run=1,
    )

    trainer.fit(dsg, datamodule=datamodule)
    trainer.save_checkpoint(f"checkpoint/{parsed_args.run_name}_{parsed_args.dataset}.ckpt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DisentanGAN training script")

    ### PARSED ARGUMENTS ###
    ## DEFAULT ARGS ##
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500000,
        help="target duration to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument("--latent", type=int, default=128, help="style space latent dimensionality")
    parser.add_argument("--image_size", type=int, default=32, help="img size to resize to")
    parser.add_argument("--n_mlp", type=int, default=8, help="number of layer in the mapping network")
    parser.add_argument("--store_images_every", type=int, default=1, help="store images per n epochs")
    parser.add_argument("--seed", type=int, default=42, help="seed to reproduce experiments")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--dataloader_workers", type=int, default=2, help="number of workers for data loaders"
    )
    parser.add_argument(
        "--classifier", type=str, default="Linear", help="classifier to be used"
    )
    parser.add_argument(
        "--classifier_classes", type=int, default=10, help="number of classes for the classifier"
    )

    ## NON-DEFAULT ARGS ##
    # arguments not contained in defaultvalues.py
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoint to resume training",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="max training epochs"
    )

    parser.add_argument("--gpu", type=str, help="GPU ID(s)")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--run_name", type=str, help="Run name")
    parser.add_argument("--dataset", type=str, help="dataset to train on")
    parser.add_argument("--logger", type=str, default="wandb", help="logger to be used")
    parsed_args = parser.parse_args()
    ### PARSED ARGUMENTS END ###

    train(parsed_args)
