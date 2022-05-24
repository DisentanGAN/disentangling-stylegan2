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

  w        z        x        y   
-----> F -----> G -----> E -----> D --> real / fake

end to end view for real images:
                             y   
                    X -> E -----> D --> real / fake

see also [here](https://arxiv.org/pdf/2004.04467.pdf),
figure 1 on page 3.


In this implementation:

MappingNetwork (F): fully connected network that translates
a meaningless random vector w into some hopefully
disentangled and thereby meaningful latent vector z

Generator (G): takes as input latent vector z and feeds it
as style vector to each upsampling layer, effectively
converting z to a representation x in pixel space

Encoder (E): takes as input an instance from pixel
space (x for synthesized images, X for real images)
and embeds it into a latent representation y, which
is being optimized to be equal to original latent
representation z

Discriminator (D): takes as input a latent embedding y
and decides whether y embeds a real or a fake image,
thus enforcing the adversarial learning scheme


TODO: interface for optional downstream tasks
TODO: Connector class optimizes y to be equal to w.

"""

import pytorch_lightning as pl

class Connector(pl.LightningModule):
    def __init__(
        self,
        mapping,
        generator,
        encoder,
        discriminator):
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

