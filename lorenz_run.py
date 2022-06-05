import pytorch_lightning as pl

from disentangledsg import DisentangledSG

from defaultvalues import default_args

from pytorch_lightning.loggers import WandbLogger

args = default_args
args.gpu=0
args.augment=False
args.d_reg_every=3


dsg = DisentangledSG(args)

wandb_logger = WandbLogger(project="disentangling-gan")

trainer = pl.Trainer(fast_dev_run=500, gpus=[0], logger=wandb_logger)
trainer.fit(dsg)