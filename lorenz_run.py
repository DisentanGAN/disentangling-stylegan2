import pytorch_lightning as pl

from disentangledsg import DisentangledSG

from defaultvalues import default_args

from pytorch_lightning.loggers import WandbLogger

args = default_args

dsg = DisentangledSG(args)

wandb_logger = WandbLogger(project="disentangling-gan")

trainer = pl.Trainer(gpus=[0], logger=wandb_logger)
trainer.fit(dsg, ckpt_path="/home/ocean/disentangling_gan/disentangling-stylegan2/disentangling-gan/10pmcl3u/checkpoints/epoch=29-step=168750.ckpt")