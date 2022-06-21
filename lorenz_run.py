import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from disentangledsg import DisentangledSG
from defaultvalues import default_args
from pytorch_lightning.callbacks import ModelCheckpoint
from datamodules import MNISTDataModule


wandb_logger = WandbLogger(project="disentangling-gan")

default_args['run_name'] = wandb_logger.version

dsg = DisentangledSG(default_args)

mnist = MNISTDataModule()

trainer = pl.Trainer(
    default_root_dir='/netscratch',
    gpus=[0],
    logger=wandb_logger,
    max_time="00:06:00:00"
    )

trainer.fit(dsg, mnist)
