
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from disentangledsg import DisentangledSG
from defaultvalues import default_args, default_args_help
from datamodules import MNISTDataModule, PCAMDataModule

def train(hparams):

    if hparams.checkpoint_path:
        dsg = DisentangledSG.load_from_checkpoint(
            checkpoint_path=hparams.checkpoint_path)
    else:
        dsg = DisentangledSG(vars(hparams))

    wandb_logger = WandbLogger(project="disentangling-gan")
    hparams.run_name = wandb_logger.version

    datasets = {'mnist': MNISTDataModule,
                'pcam': PCAMDataModule}
    datamodule = datasets[hparams.dataset]()

    trainer = pl.Trainer(
        ## DEVICE ##
        gpus=hparams.gpu,
        # strategy='ddp',
        # LOGGING AND CALLBACKS
        # callbacks=[],
        logger=wandb_logger,
        ## TRAIN DURATION ##
        #max_epochs=hparams.max_epochs,
        max_time=hparams.max_time,
        # max_time="00:12:00:00",
        ## DEBUG OPTIONS ##
        # fast_dev_run=1,
    )

    trainer.fit(dsg, datamodule=datamodule)
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
        #elif type(dvalue) == bool and dvalue == True:
        #    parser.add_argument(
        #        f"--{key}",
        #        action="store_false",
        #        help=default_args_help[key]
        #    )
        # store other values
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
    #parser.add_argument("--logger", type=str, default="wandb", help="logger to be used")

if __name__ == "__main__":
    parser = setup_default_parser_args()
    setup_non_default_args(parser)
    hparams = parser.parse_args()

    #train(hparams)
