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

  z        w        x        y       d   
-----> F -----> G -----> E -----> D ---> real / fake

end to end view for real images:
                             y       d
                    X -> E -----> D ---> real / fake

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

from torch.nn import ModuleList

from defaultvalues import optim_conf
from util import requires_grad, mixing_noise
from non_leaking import augment, AdaptiveAugment




class DisentangledSG(pl.LightningModule):
    def __init__(
        self,
        mappingnetwork,
        generator,
        encoder,
        discriminator,
        optim_conf=optim_conf):

        super().__init__()

        self.mapping = mappingnetwork
        self.generator = generator
        self.encoder = encoder
        self.discriminator = discriminator

        self.submodules = [
                self.mapping,
                self.generator,
                self.encoder,
                self.discriminator,
        ]

        self.latent  = mappingnetwork.style_dim
        self.imgsize = generator.size

        self.augment_data = False

        # TODO: flexible downstream tasks (as a list)
        #self.downstream = None

        self.optim_conf = optim_conf


    def forward(self, z):
        w = self.mapping(z)
        x = self.generator([w])
        y = self.encoder(x[0])
        d = self.discriminator(y)

        # TODO: what is the latent value in x[1]?

        # TODO: how to insert downstream tasks here?
        return d


    def set_trainable(self, *trainable_models):
        """
        freeze all parameters of all models,
        then unfreeze parameters of models
        in argument for training.
        """
        for i in self.submodules:
            requires_grad(i, flag=False)
        for i in trainable_models:
            requires_grad(i, flag=True)


    def training_step(self, batch, batch_idx, optimizer_idx):
        # batch contains x, y (data and label)
        # configure pytorch lightning for multistage training steps:
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html?highlight=training_step#training-step

        if optimizer_idx == 0:
            # OPTIMIZE DISCRIMINATION
    
            self.set_trainable(
                    self.discriminator,
                    self.encoder)

            real_img_batch = batch[0]
            batch_size = real_img_batch.shape[0]

            noise = mixing_noise(batch_size, self.latent, args.mixing)
            fake_img_batch, _ = self.generator([mapping(z) for z in noise])

            if self.augment_data:
                real_img_aug, _ = augment(real_img_batch, ada_aug_p)
                fake_img, _ = augment(fake_img_batch, ada_aug_p)

            else:
                real_img_aug = real_img_batch


        if optimizer_idx == 1:
            # OPTIMIZE GENERATION
            self.set_trainable(
                    self.mapping,
                    self.generator)
            pass

        if optimizer_idx == 2:
            # OPTIMIZE CONSISTENCY
            self.set_trainable(
                    self.mapping,
                    self.generator,
                    self.encoder)
            pass



    def configure_optimizers(self):

        # for the following tasks there will be separate
        # optimizations and corresponding partial 
        # training steps 
        tasks = ["discrimination", "generation", "consistency"]

        # select for each task the corresponding 
        # submodules to optimize
        params = [ModuleList(self.encoder, self.discriminator),
                ModuleList(self.mapping, self.generator),
                ModuleList(self.mapping, self.generator, self.encoder)]
        # shorthand
        o = self.optim_conf

        # generate an optimizer for each task according
        # to optimizer configuration in variable optim_conf
        # (default_values.py:26)
        optim = [o[i]["optimizer"](j.parameters(), **o[i]["args"]) \
                for i, j in zip(tasks, params)]

        # return a list of optimizers
        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.configure_optimizers
        return optim

