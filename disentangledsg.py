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
import torch

from torch.nn import ModuleList
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from defaultvalues import optim_conf, default_args
from util import (
    requires_grad,
    mixing_noise,
    d_logistic_loss,
    d_r1_loss,
    g_nonsaturating_loss,
    g_path_regularize,
)
from non_leaking import augment, AdaptiveAugment

from mappingnetwork import MappingNetwork
from generator import Generator
from discriminator import Discriminator
from encoder import Encoder


class DisentangledSG(pl.LightningModule):
    def __init__(
        self,
        args=default_args,
        optim_conf=optim_conf):

        super().__init__()

        self.save_hyperparameters()
        self.args = args

        self.automatic_optimization = False

        pl.seed_everything(self.args['seed'])

        self.mapping = MappingNetwork(args.['latent'], args.['n_mlp'])
        self.generator = Generator(args.['image_size'], args.['latent'])
        self.encoder = Encoder(args.['image_size'], args.['latent'])
        self.discriminator = Discriminator(args.['latent'])

        self.submodules = [
                self.mapping,
                self.generator,
                self.encoder,
                self.discriminator,
        ]

        self.ada_augment = AdaptiveAugment(args.['ada_target'], args.['ada_length'], 8, self.device)


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


    def training_step(self, batch, batch_idx):
        self.optimize_discrimination(batch, batch_idx)
        self.optimize_generation(batch, batch_idx)
        self.optimize_consistency(batch)
        
    def on_epoch_start(self) -> None:

        if self.example_data['images'][0].device.type == 'cpu':
            self.example_data['images'] = [image.to(self.device) for image in self.example_data['images']]
            self.example_data['noise'] = [noise.to(self.device) for noise in self.example_data['noise']]
            self.example_data['z'] = [z.to(self.device) for z in self.example_data['z']]

        if self.current_epoch % self.args['store_images_every'] == 0:
            self.log_images()

    def log_images(self) -> None:
        pass
        if type(self.logger) == pl.loggers.wandb.WandbLogger: 
            self.logger.log_image(key='original images', images = self.example_data['images'])

        # reconstruct real images
        w = self.encoder(torch.stack(self.example_data['images']))
        images, _ = self.generator([w], noise=self.example_data['noise'])

        if type(self.logger) == pl.loggers.wandb.WandbLogger: 
            self.logger.log_image(key='original images - reconstructed', images = list(images))

        # generate from noise
        w = self.mapping(torch.stack(self.example_data['z'])[0])
        images, _ = self.generator([w], noise=self.example_data['noise'])

        if type(self.logger) == pl.loggers.wandb.WandbLogger: 
            self.logger.log_image(key='synthetic images', images = list(images))

        # reconstruct from noise
        w = self.encoder(images)
        images, _ = self.generator([w], noise=self.example_data['noise'])

        if type(self.logger) == pl.loggers.wandb.WandbLogger: 
            self.logger.log_image(key='synthetic images - reconstructed', images = list(images))

    def optimize_discrimination(self, batch, batch_idx):
        # check for regularisation interval
        d_regularize = batch_idx % self.args['d_reg_every'] == 0
        if d_regularize:
             return self.regularize_discrimination(batch)
        # default: do normal optimisation
        self.optimize_enc_and_disc(batch)


    def optimize_enc_and_disc(self, batch):
        _, discriminator_optimizer, encoder_optimizer, _ = self.optimizers()
        discriminator_optimizer.zero_grad()
        encoder_optimizer.zero_grad()

        self.set_trainable(
                self.encoder,
                self.discriminator)

        real_img = batch[0]
        batch_size = real_img.shape[0]

        noise = mixing_noise(batch_size, self.args['latent'], self.args['mixing'], self.device)
        fake_img, _ = self.generator([self.mapping(z) for z in noise])

        if self.args['augment']:
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

        self.log('discriminator/loss', d_loss)
        self.log('discriminator/real_score', real_pred.mean())
        self.log('discriminator/fake_score', fake_pred.mean())

        if self.args['augment'] and self.args['augment_p'] == 0:
            self.ada_augment.tune(real_pred)

        self.manual_backward(d_loss)
        discriminator_optimizer.step()
        encoder_optimizer.step()



    def regularize_discrimination(self, batch):
        _, discriminator_optimizer, encoder_optimizer, _ = self.optimizers()
        discriminator_optimizer.zero_grad()
        encoder_optimizer.zero_grad()

        self.set_trainable(
                self.encoder,
                self.discriminator)

        real_img = batch[0]
        real_img.requires_grad = True

        if self.args['augment']:
            real_img_aug, _ = augment(real_img, self.ada_augment.ada_aug_p)

        else:
            real_img_aug = real_img

        real_pred = self.discriminator(self.encoder(real_img_aug))
        r1_loss = d_r1_loss(real_pred, real_img)

        reg_loss = (self.args['r1'] / 2 * r1_loss * self.args['d_reg_every'] + 0 * real_pred[0])[0]
        self.loss_dict["r1"] = r1_loss

        self.log('r1/loss', r1_loss)
        self.log('regularization/loss', reg_loss)

        self.manual_backward(reg_loss)
        discriminator_optimizer.step()
        encoder_optimizer.step()


    def optimize_generation(self, batch, batch_idx):
        # check for regularisation interval
        g_regularize = batch_idx % self.args['g_reg_every'] == 0
        if g_regularize:
            return self.regularize_generation(batch)
        # default: do normal optimisation
        return self.optimize_map_and_gen(batch)


    def optimize_map_and_gen(self, batch):
        generator_optimizer, _, _, mapping_optimizer = self.optimizers()
        generator_optimizer.zero_grad()
        mapping_optimizer.zero_grad()

        self.set_trainable(
                self.mapping,
                self.generator)

        real_img   = batch[0]
        batch_size = real_img.shape[0]

        noise = mixing_noise(batch_size, self.args['latent'], self.args['mixing'], self.device)
        fake_img, _ = self.generator([self.mapping(z) for z in noise])

        if self.args['augment']:
            fake_img, _ = augment(fake_img, self.ada_augment.ada_aug_p)

        fake_pred = self.discriminator(self.encoder(fake_img))
        g_loss = g_nonsaturating_loss(fake_pred)

        self.loss_dict["g"] = g_loss

        self.log('generator/loss', g_loss)

        self.manual_backward(g_loss)
        generator_optimizer.step()
        mapping_optimizer.step()

    def regularize_generation(self, batch):
        generator_optimizer, _, _, mapping_optimizer = self.optimizers()
        generator_optimizer.zero_grad()
        mapping_optimizer.zero_grad()

        self.set_trainable(
                self.mapping,
                self.generator)

        batch_size = batch[0].shape[0]

        path_batch_size = max(1, batch_size // self.args['path_batch_shrink'])
        noise = mixing_noise(path_batch_size, self.args['latent'], self.args['mixing'], self.device)
        fake_img, latents = self.generator([self.mapping(z) \
                for z in noise], return_latents=True)

        path_loss, self.mean_path_length, path_lengths = g_path_regularize(
            fake_img, latents, self.mean_path_length
        )

        weighted_path_loss = self.args['path_regularize'] * self.args['g_reg_every'] * path_loss
        if self.args['path_batch_shrink']:
            weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

        self.loss_dict["path"] = path_loss
        self.loss_dict["path_length"] = path_lengths.mean()

        self.log('path/loss', path_loss)
        self.log('path/length', path_lengths.mean())
        self.log('path/weighted_path_loss', weighted_path_loss)

        self.manual_backward(weighted_path_loss)
        generator_optimizer.step()
        mapping_optimizer.step()


    def optimize_consistency(self, batch):
        generator_optimizer, _, encoder_optimizer, mapping_optimizer = self.optimizers()
        generator_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        mapping_optimizer.zero_grad()

        real_img   = batch[0]
        batch_size = real_img.shape[0]

        self.set_trainable(
                self.generator,
                self.mapping,
                self.encoder)

        batch_size = batch[0].shape[0]

        w_z = self.mapping(torch.randn(batch_size, self.args['latent'], device=self.device))
        fake_img, _ = self.generator([w_z])

        consistency_loss = (w_z - self.encoder(fake_img)).pow(2).mean()
        self.loss_dict["consistency_loss"] = consistency_loss

        self.log('consistency/loss', consistency_loss)

        self.manual_backward(consistency_loss)
        generator_optimizer.step()
        encoder_optimizer.step()
        mapping_optimizer.step()

    def configure_optimizers(self):

        generator_optimizer = self.optim_conf['generator']['optimizer'](
            self.generator.parameters(),
            lr = self.optim_conf['generator']['args']['lr']
            )
        discriminator_optimizer = self.optim_conf['discriminator']['optimizer'](
            self.discriminator.parameters(),
            lr = self.optim_conf['discriminator']['args']['lr']
            )
        encoder_optimizer = self.optim_conf['encoder']['optimizer'](
            self.encoder.parameters(),
            lr = self.optim_conf['encoder']['args']['lr']
            )
        mapping_optimizer = self.optim_conf['mapping']['optimizer'](
            self.mapping.parameters(),
            lr = self.optim_conf['mapping']['args']['lr']
            )
        
        return [generator_optimizer, discriminator_optimizer, encoder_optimizer, mapping_optimizer]

    def prepare_data(self):
        transform = transforms.Compose([transforms.Grayscale(3), transforms.Resize(32), transforms.ToTensor()])

        self.training_data = MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform
        )

        self.example_data = {
            'images': [self.training_data[i][0] for i in range(self.args['num_example_images'])],
            'noise': self.generator.make_noise(),
            'z': [torch.randn(self.args['num_example_images'], self.args['latent'])]
        }

    def train_dataloader(self):
        return DataLoader(
            self.training_data,                       
            batch_size=self.args['batch_size']
        )
