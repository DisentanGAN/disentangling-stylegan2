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
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from classifier import LinearClassifier
from defaultvalues import channels, default_args, optim_conf
from discriminator import Discriminator
from encoder import Encoder
from generator import Generator
from mappingnetwork import MappingNetwork
from non_leaking import AdaptiveAugment, augment
from util import (d_logistic_loss, d_r1_loss, g_nonsaturating_loss,
                  g_path_regularize, mixing_noise, requires_grad)


class DisentangledSG(pl.LightningModule):
    def __init__(
            self,
            args=default_args,
            optim_conf=optim_conf,
            channels=channels):

        super().__init__()

        self.save_hyperparameters()
        self.args = args

        self.automatic_optimization = False

        pl.seed_everything(self.args['seed'])

        self.mapping = MappingNetwork(args['latent'], args['n_mlp'])
        self.generator = Generator(
            args['image_size'], args['latent'], channels)
        self.encoder = Encoder(args['image_size'], args['latent'], channels)
        self.discriminator = Discriminator(args['latent'])

        self.submodules = [
            self.mapping,
            self.generator,
            self.encoder,
            self.discriminator,
        ]

        if self.args['classifier'] == 'Linear':
            self.classifier = LinearClassifier(
                self.args['latent'], self.args['classifier_classes'])
            self.classifier_loss = torch.nn.CrossEntropyLoss()
        elif self.args['classifier'] == 'Resnet':
            self.classifier = None
            self.classifier_loss = torch.nn.CrossEntropyLoss()
        else:
            self.classifier = None

        if self.classifier:
            self.submodules.append(self.classifier)

        self.ada_augment = AdaptiveAugment(
            args['ada_target'], args['ada_length'], 8, self.device)

        self.mean_path_length = 0

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
        if self.classifier:
            self.optimize_classification(batch)
    
    
    def validation_step(self, batch, batch_idx):
        if not self.classifier:
            return
        
        with torch.no_grad():

            images, labels = batch
            w = self.encoder(images)
            predicted_labels = self.classifier(w)

            classification_loss = self.classifier_loss(predicted_labels, labels)
            accuracy =  torch.sum(predicted_labels.argmax(dim=1) == labels)

            self.log('classifier/validation/loss', classification_loss)
            self.log('classifier/validation/accuracy', accuracy)

        
    def on_train_epoch_start(self) -> None:
        # self.check_seperability()

        if self.example_data['images'][0].device.type == 'cpu':
            self.example_data['images'] = [
                image.to(self.device) for image in self.example_data['images']]
            self.example_data['noise'] = [
                noise.to(self.device) for noise in self.example_data['noise']]
            self.example_data['z'] = [z.to(self.device)
                                      for z in self.example_data['z']]

        if self.current_epoch % self.args['store_images_every'] == 0:
            self.log_images()

    def log_images(self) -> None:
        if type(self.logger) == pl.loggers.wandb.WandbLogger:
            self.logger.log_image(key='original images',
                                  images=self.example_data['images'])

        # reconstruct real images
        w = self.encoder(torch.stack(self.example_data['images']))
        images, _ = self.generator([w], noise=self.example_data['noise'])

        if type(self.logger) == pl.loggers.wandb.WandbLogger:
            self.logger.log_image(
                key='original images - reconstructed', images=list(images))

        # generate from noise
        w = self.mapping(torch.stack(self.example_data['z'])[0])
        images, _ = self.generator([w], noise=self.example_data['noise'])

        if type(self.logger) == pl.loggers.wandb.WandbLogger:
            self.logger.log_image(key='synthetic images', images=list(images))

        # reconstruct from noise
        w = self.encoder(images)
        images, _ = self.generator([w], noise=self.example_data['noise'])

        if type(self.logger) == pl.loggers.wandb.WandbLogger:
            self.logger.log_image(
                key='synthetic images - reconstructed', images=list(images))

    # def check_seperability(self):
    #     colors = ['red', 'green', 'blue', 'gray', 'yellow', 'cyan', 'orange', 'black', 'purple', 'greenyellow']

    #     loader = DataLoader(
    #             self.training_data,
    #             batch_size=self.args['batch_size']
    #         )
    #     with torch.no_grad():
    #         latent_representations = []
    #         labels = []
    #         for img, label in loader:
    #             img = img.to(self.device)
    #             latent_representations.append(self.encoder(img))
    #             labels.append(label)
    #         w = torch.concat(latent_representations, dim=0).cpu().numpy()
    #         label = torch.concat(labels, dim=0).numpy()
    #         print('UMAP')
    #         reducer = umap.UMAP()
    #         print('UMAP 1')
    #         embedding = reducer.fit_transform(w[:1000])
    #         print('UMAP 2')
    #         c = [colors[i] for i in label[:1000]]
    #         print('UMAP 3')
    #         plt.scatter(embedding[:100, 0], embedding[:100, 1], c=c)
    #         print('UMAP 4')
    #         plt.gca().set_aspect('equal', 'datalim')
    #         plt.show()
    #         print('UMAP 5')
    #         # self.log('umap', plt)

    def optimize_discrimination(self, batch, batch_idx):
        # default: do normal optimisation
        self.optimize_enc_and_disc(batch)

        # check for regularisation interval
        d_regularize = batch_idx % self.args['d_reg_every'] == 0
        if d_regularize:
            self.regularize_discrimination(batch)

    def optimize_enc_and_disc(self, batch):
        discriminator_optimizer = self.optimizers[self.discriminator]
        encoder_optimizer = self.optimizers[self.encoder]

        self.set_trainable(
            self.encoder,
            self.discriminator
        )

        real_img = batch[0]
        batch_size = real_img.shape[0]

        noise = mixing_noise(
            batch_size, self.args['latent'], self.args['mixing'], self.device)
        fake_img, _ = self.generator([self.mapping(z) for z in noise])

        if self.args['augment']:
            real_img_aug, _ = augment(real_img, self.ada_augment.ada_aug_p)
            fake_img, _ = augment(fake_img, self.ada_augment.ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = self.discriminator(self.encoder(fake_img))
        real_pred = self.discriminator(self.encoder(real_img_aug))
        d_loss = d_logistic_loss(real_pred, fake_pred)

        self.log('discriminator/loss', d_loss)
        self.log('discriminator/real_score', real_pred.mean())
        self.log('discriminator/fake_score', fake_pred.mean())

        if self.args['augment'] and self.args['augment_p'] == 0:
            self.ada_augment.tune(real_pred)

        discriminator_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        self.manual_backward(d_loss)
        discriminator_optimizer.step()
        encoder_optimizer.step()

    def regularize_discrimination(self, batch):

        discriminator_optimizer = self.optimizers[self.discriminator]
        encoder_optimizer = self.optimizers[self.encoder]

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

        reg_loss = (self.args['r1'] / 2 * r1_loss *
                    self.args['d_reg_every'] + 0 * real_pred[0])[0]

        self.log('r1/loss', r1_loss)
        self.log('regularization/loss', reg_loss)

        discriminator_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        self.manual_backward(reg_loss)
        discriminator_optimizer.step()
        encoder_optimizer.step()

    def optimize_generation(self, batch, batch_idx):
        # default: do normal optimisation
        self.optimize_map_and_gen(batch)

        # check for regularisation interval
        g_regularize = batch_idx % self.args['g_reg_every'] == 0
        if g_regularize:
            self.regularize_generation(batch)

    def optimize_map_and_gen(self, batch):

        generator_optimizer = self.optimizers[self.generator]
        mapping_optimizer = self.optimizers[self.mapping]

        self.set_trainable(
            self.mapping,
            self.generator)

        real_img = batch[0]
        batch_size = real_img.shape[0]

        noise = mixing_noise(
            batch_size, self.args['latent'], self.args['mixing'], self.device)
        fake_img, _ = self.generator([self.mapping(z) for z in noise])

        if self.args['augment']:
            fake_img, _ = augment(fake_img, self.ada_augment.ada_aug_p)

        fake_pred = self.discriminator(self.encoder(fake_img))
        g_loss = g_nonsaturating_loss(fake_pred)

        self.log('generator/loss', g_loss)

        generator_optimizer.zero_grad()
        mapping_optimizer.zero_grad()
        self.manual_backward(g_loss)
        generator_optimizer.step()
        mapping_optimizer.step()

    def regularize_generation(self, batch):
        generator_optimizer = self.optimizers[self.generator]
        mapping_optimizer = self.optimizers[self.mapping]

        self.set_trainable(
            self.mapping,
            self.generator)

        batch_size = batch[0].shape[0]

        path_batch_size = max(1, batch_size // self.args['path_batch_shrink'])
        noise = mixing_noise(
            path_batch_size, self.args['latent'], self.args['mixing'], self.device)
        fake_img, latents = self.generator(
            [self.mapping(z) for z in noise],
            return_latents=True
        )

        path_loss, self.mean_path_length, path_lengths = g_path_regularize(
            fake_img, latents, self.mean_path_length
        )

        weighted_path_loss = self.args['path_regularize'] * \
            self.args['g_reg_every'] * path_loss
        if self.args['path_batch_shrink']:
            weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

        self.log('path/loss', path_loss)
        self.log('path/length', path_lengths.mean())
        self.log('path/weighted_path_loss', weighted_path_loss)

        generator_optimizer.zero_grad()
        mapping_optimizer.zero_grad()
        self.manual_backward(weighted_path_loss)
        generator_optimizer.step()
        mapping_optimizer.step()

    def optimize_consistency(self, batch):
        generator_optimizer = self.optimizers[self.generator]
        encoder_optimizer = self.optimizers[self.encoder]
        mapping_optimizer = self.optimizers[self.mapping]

        real_img = batch[0]
        batch_size = real_img.shape[0]

        self.set_trainable(
            self.generator,
            self.mapping,
            self.encoder)

        batch_size = batch[0].shape[0]

        w_z = self.mapping(torch.randn(
            batch_size, self.args['latent'], device=self.device))
        fake_img, _ = self.generator([w_z])

        consistency_loss = (w_z - self.encoder(fake_img)).pow(2).mean()

        self.log('consistency/loss', consistency_loss)

        generator_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        mapping_optimizer.zero_grad()
        self.manual_backward(consistency_loss)
        generator_optimizer.step()
        encoder_optimizer.step()
        mapping_optimizer.step()

    def optimize_classification(self, batch):
        encoder_optimizer = self.optimizers[self.encoder]
        classifier_optimizer = self.optimizers[self.classifier]

        self.set_trainable(
            self.encoder,
            self.classifier)

        images, labels = batch
        w = self.encoder(images)
        predicted_labels = self.classifier(w).argmax(dim=1)
        classification_loss = self.classifier_loss(predicted_labels, labels)
        accuracy =  torch.sum(predicted_labels == labels)

        self.log('classifier/train/loss', classification_loss)
        self.log('classifier/train/accuracy', accuracy)

        encoder_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        self.manual_backward(classification_loss)
        encoder_optimizer.step()
        classifier_optimizer.step()

    def configure_optimizers(self):

        generation_regularization_ratio = self.args['d_reg_every'] / (
            self.args['d_reg_every'] + 1)
        discrimination_regularization_ratio = self.args['d_reg_every'] / (
            self.args['d_reg_every'] + 1)

        generator_optimizer = self.optim_conf['generator']['optimizer'](
            self.generator.parameters(),
            lr=self.optim_conf['generator']['args']['lr'] *
            generation_regularization_ratio,
            betas=(0, 0.99 ** generation_regularization_ratio)
        )
        discriminator_optimizer = self.optim_conf['discriminator']['optimizer'](
            self.discriminator.parameters(),
            lr=self.optim_conf['discriminator']['args']['lr'] *
            discrimination_regularization_ratio,
            betas=(0, 0.99 ** discrimination_regularization_ratio)
        )
        encoder_optimizer = self.optim_conf['encoder']['optimizer'](
            self.encoder.parameters(),
            lr=self.optim_conf['encoder']['args']['lr'] *
            discrimination_regularization_ratio,
            betas=(0, 0.99 ** discrimination_regularization_ratio)
        )
        mapping_optimizer = self.optim_conf['mapping']['optimizer'](
            self.mapping.parameters(),
            lr=self.optim_conf['mapping']['args']['lr'] *
            generation_regularization_ratio,
            betas=(0, 0.99 ** generation_regularization_ratio)
        )

        optimizer = [generator_optimizer, discriminator_optimizer,
                     encoder_optimizer, mapping_optimizer]
        self.optimizers = {
            self.mapping: mapping_optimizer,
            self.encoder: encoder_optimizer,
            self.generator: generator_optimizer,
            self.discriminator: discriminator_optimizer
        }

        if self.classifier:
            classifier_optimizer = self.optim_conf['classifier']['optimizer'](
                self.classifier.parameters(),
                lr=self.optim_conf['classifier']['args']['lr']
            )
            optimizer.append(classifier_optimizer)
            self.optimizers[self.classifier] = classifier_optimizer

        return optimizer

    def prepare_data(self):
        transform = transforms.Compose([
            # make MNIST images 32x32x3
            transforms.Grayscale(3),
            transforms.Pad(2),

            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,), inplace=True),
        ])

        self.training_data = MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform
        )

        self.validation_data = MNIST(
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
            batch_size=self.args['batch_size'],
            num_workers=self.args['dataloader_workers']
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_data,
            batch_size=self.args['batch_size'],
            num_workers=self.args['dataloader_workers']
        )
