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
from util import (
    requires_grad,
    mixing_noise,
    d_logistic_loss,
    d_r1_loss,
    g_nonsaturating_loss,
)
from non_leaking import augment, AdaptiveAugment





class DisentangledSG(pl.LightningModule):
    def __init__(
        self,
        mappingnetwork,
        generator,
        encoder,
        discriminator,
        args,
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

        device = f"cuda:{args.gpu}"
        self.ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

        self.imgsize = generator.size

        self.args = args

        self.mean_path_length = 0

        self.loss_dict = {}

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

        if optimizer_idx == 0:
            return self.optimize_discrimination(batch, batch_idx)
        if optimizer_idx == 1:
            return self.optimize_generation(batch, batch_idx)
        if optimizer_idx == 2:
            return self.optimize_consistency(batch, batch_idx)

        # we should never arrive here
        return None


    def optimize_discrimination(self, batch, batch_idx):
        # check for regularisation interval
        d_regularize = batch_idx % self.args.d_reg_every == 0
        if d_regularize:
            return self.regularize_discrimination(batch)
        # default: do normal optimisation
        return self.optimize_enc_and_disc(batch)


    def optimize_enc_and_disc(self, batch):
        self.set_trainable(
                self.encoder,
                self.discriminator)

        real_img   = batch[0]
        batch_size = real_img.shape[0]

        noise = mixing_noise(batch_size, self.args.latent, self.args.mixing)
        fake_img, _ = self.generator([self.mapping(z) for z in noise])

        if self.args.augment:
            real_img_aug, _ = augment(real_img, self.ada_augment.ada_aug_p)
            fake_img, _ = augment(fake_img, self.ada_augment.ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = self.discriminator(self.encoder(fake_img))
        real_pred = self.discriminator(self.encoder(real_img_aug))
        d_loss = d_logistic_loss(real_pred, fake_pred)

        self.loss_dict["d"] = d_loss
        self.loss_dict["real_score"] = real_pred.mean()
        self.loss_dict["fake_score"] = fake_pred.mean()

        if self.args.augment and self.args.augment_p == 0:
            self.ada_augment.tune(real_pred)

        return {"loss": d_loss}


    def regularize_discrimination(self, batch):
        real_img = batch[0]
        self.set_trainable(
                self.encoder,
                self.discriminator,
                real_img)

        if self.args.augment:
            real_img_aug, _ = augment(real_img, self.ada_augment.ada_aug_p)

        else:
            real_img_aug = real_img

        real_pred = self.discriminator(self.encoder(real_img_aug))
        r1_loss = d_r1_loss(real_pred, real_img)

        reg_loss = (self.args.r1 / 2 * r1_loss * self.args.d_reg_every + 0 * real_pred[0])
        self.loss_dict["r1"] = r1_loss

        return {"loss": reg_loss}


    def optimize_generation(self, batch, batch_idx):
        # check for regularisation interval
        g_regularize = batch_idx % self.args.g_reg_every == 0
        if g_regularize:
            return self.regularize_generation(batch)
        # default: do normal optimisation
        return self.optimize_map_and_gen(batch)


    def optimize_map_and_gen(self, batch):
        self.set_trainable(
                self.mapping,
                self.generator)

        real_img   = batch[0]
        batch_size = real_img.shape[0]

        noise = mixing_noise(batch_size, self.args.latent, self.args.mixing)
        fake_img, _ = self.generator([self.mapping(z) for z in noise])

        if self.args.augment:
            fake_img, _ = augment(fake_img, self.ada_augment.ada_aug_p)

        fake_pred = self.discriminator(self.encoder(fake_img))
        g_loss = g_nonsaturating_loss(fake_pred)

        self.loss_dict["g"] = g_loss

        return {"loss": g_loss}


    def regularize_generation(self, batch, batch_idx):

        batch_size = batch[0].shape[0]

        path_batch_size = max(1, batch_size // self.args.path_batch_shrink)
        noise = mixing_noise(path_batch_size, self.args.latent, self.args.mixing)
        fake_img, latents = self.generator([self.mapping(z) \
                for z in noise], return_latents=True)

        path_loss, self.mean_path_length, path_lengths = g_path_regularize(
            fake_img, latents, self.mean_path_length
        )

        weighted_path_loss = self.args.path_regularize * self.args.g_reg_every * path_loss
        if self.args.path_batch_shrink:
            weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

        self.loss_dict["path"] = path_loss
        self.loss_dict["path_length"] = path_lengths.mean()

        return {"loss": weighted_path_loss}


    def optimize_consistency(self, batch, batch_idx):

        real_img   = batch[0]
        batch_size = real_img.shape[0]

        self.set_trainable(
                self.generator,
                self.mapping,
                self.encoder)

        batch_size = batch[0].shape[0]

        w_z = self.mapping(torch.randn(batch_size, self.args.latent))
        fake_img, _ = self.generator([w_z])

        consistency_loss = (w_z - self.encoder(fake_img)).pow(2).mean()
        self.loss_dict["consistency_loss"] = consistency_loss

        return {"loss": consistency_loss}


    def configure_optimizers(self):

        # for the following tasks there will be separate
        # optimizations and corresponding partial 
        # training steps 
        tasks = ["discrimination", "generation", "consistency"]

        # select for each task the corresponding 
        # submodules to optimize
        params = [ModuleList([self.encoder, self.discriminator]),
                ModuleList([self.mapping, self.generator]),
                ModuleList([self.mapping, self.generator, self.encoder])]
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

