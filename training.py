
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from disentangledsg import DisentangledSG
from defaultvalues import default_args, default_args_help
from datamodules import MNISTDataModule, PCAMDataModule

def train(hparams):

    dsg = DisentangledSG(vars(hparams))

    wandb_logger = WandbLogger(project="disentangling-gan")

    if hparams.run_name is None:
        hparams.run_name = wandb_logger.version

    datasets = {'mnist': MNISTDataModule,
                'pcam': PCAMDataModule}
    datamodule = datasets[hparams.dataset]()
    
    trainer = pl.Trainer(
        ## DEVICE ##
        gpus=hparams.gpu,
        logger=wandb_logger,
        ## TRAIN DURATION ##
        max_epochs=hparams.max_epochs,
        max_time=hparams.max_time,
        ## DEBUG OPTIONS ##
        # fast_dev_run=1,
    )

    if hparams.debug:    
        trainer = pl.Trainer(
            ## DEVICE ##
            gpus=hparams.gpu,
            logger=wandb_logger,
            limit_train_batches=0.01,
            limit_val_batches=0.01)         
    
    trainer.fit(dsg, datamodule=datamodule, ckpt_path=hparams.ckpt)
    trainer.save_checkpoint(
        f"checkpoint/{hparams.run_name}_{hparams.dataset}.ckpt")

def setup_default_parser_args():
    parser = argparse.ArgumentParser(description="DisentanGAN training script")

    for key in default_args.keys():
        dvalue = default_args[key]
        # store bool values properly
        if type(dvalue) == bool:# and dvalue == False:
            parser.add_argument(
                f"--{key}",
                action="store_true" if dvalue == False else "store_false",
                help=default_args_help[key]
            )
        else:
            parser.add_argument(
                f"--{key}",
                type=type(default_args[key]),
                default=default_args[key],
                help=default_args_help[key]
            )

    return parser

def setup_non_default_args(parser):
    parser.add_argument("--max_epochs", type=int, default=100, help="max training epochs")
    parser.add_argument("--max_time", type=str, default="00:04:00:00", help="max training time")
    parser.add_argument("--gpu", type=str, default="1", help="GPU ID(s)")
    parser.add_argument("--run_name", type=str, default=None, help="Run name")
    parser.add_argument("--dataset", type=str, help="dataset to train on")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint to resume training from")
    parser.add_argument("--debug", type=bool, default=False, help="run in debug mode")
    #parser.add_argument("--logger", type=str, default="wandb", help="logger to be used")

if __name__ == "__main__":
    parser = setup_default_parser_args()
    setup_non_default_args(parser)
    hparams = parser.parse_args()

    train(hparams)
