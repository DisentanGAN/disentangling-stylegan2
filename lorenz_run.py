import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from disentangledsg import DisentangledSG
from defaultvalues import default_args
from datamodules import MNISTDataModule

dsg = DisentangledSG(default_args)

#wandb_logger = WandbLogger(project="disentangling-gan")

mnist = MNISTDataModule()

trainer = pl.Trainer(
    #default_root_dir='/netscratch',
    gpus=[0],
    #logger=wandb_logger,
    max_epochs=200)

trainer.fit(dsg, mnist)
