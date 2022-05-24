import pytorch_lightning as pl

from util import *


"""
TODO: the Discriminator only
does the classification on the
latent embedding y, which was
produced by the Encoder.

Architecture of Discriminator?
Probably a single or a few
fully connected layers?

Input is y, output is true/false
"""

class Discriminator(pl.LightningModule):
    def __init__(self):
        pass

    def forward(self):
        pass

    def training_step(self):
        pass

    def test_step(self):
        pass

    def predict_step(self):
        pass

    def configure_optimizers(self):
        pass

