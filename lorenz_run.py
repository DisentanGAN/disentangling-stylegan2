import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from disentangledsg import DisentangledSG
from defaultvalues import default_args

print('start')
dsg = DisentangledSG(default_args)

wandb_logger = WandbLogger(project="disentangling-gan")

trainer = pl.Trainer(
    default_root_dir='/netscratch',
    gpus=[0],
    logger=wandb_logger,
    max_epochs=200)

trainer.fit(dsg, ckpt_path='/netscratch/disentangling-gan/3f31ny4b/checkpoints/epoch=117-step=1137117.ckpt')