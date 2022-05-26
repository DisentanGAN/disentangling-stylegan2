"""
The Connector class orchestrates the
styleGAN training scheme by connecting
the different components.

In particular, the training scheme is
extended to force the discriminator
towards embedding any seen image to
its corresponding latent vector w before
classification by adding the difference
from the discriminator output y to w to
the loss function.

Further downstream tasks on y should help
to disentangle y and thereby w.



end to end view for synthesized images:

  z        w        x        y   
-----> F -----> G -----> E -----> D --> real / fake

end to end view for real images:
                             y   
                    X -> E -----> D --> real / fake

see also [here](https://arxiv.org/pdf/2004.04467.pdf),
figure 1 on page 3.


In this implementation:

MappingNetwork (F): fully connected network that translates
a meaningless random vector z into some hopefully
disentangled and thereby meaningful latent vector w

Generator (G): takes as input latent vector w and feeds it
as style vector to each upsampling layer, effectively
converting w to a representation x in pixel space

Encoder (E): takes as input an instance from pixel
space (x for synthesized images, X for real images)
and embeds it into a latent representation y, which
is being optimized to be equal to original latent
representation w

Discriminator (D): takes as input a latent embedding y
and decides whether y embeds a real or a fake image,
thus enforcing the adversarial learning scheme


TODO: interface for optional downstream tasks
TODO: Connector class optimizes y to be equal to w.

"""

import pytorch_lightning as pl

from defaultvalues import optimizers




class DisentangledSG(pl.LightningModule):
    def __init__(
        self,
        mappingnetwork,
        generator,
        encoder,
        discriminator,
        optimizers=optimizers):

        super().__init__()

        self.mapping = mappingnetwork
        self.generator = generator
        self.encoder = encoder
        self.discriminator = discriminator

        #self.downstream = None



        self.optimizers = optimizers


    def forward(self):
        pass

    def training_step(self):
        pass

    def test_step(self):
        pass

    def predict_step(self):
        pass

    def configure_optimizers(self):
        optim = []
        for i in [self.mappingnetwork, 
                self.generator,
                self.encoder,
                self.discriminator]:

            if i.module_name() in self.optimizers:
                # get specified optimizer options for each submodule
                entry = self.optimizer[i.module_name()]
            else:
                # no optimizer options specified: apply default
                entry = self.optimizer["default"]

            optim.append(entry["optimizer"](i.parameters(), **entry["args"]))

        return tuple(optim)

