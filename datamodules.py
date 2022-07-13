"""
Contains LightningDataModules for supported datasets.
"""
from typing import Optional

from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST, PCAM

import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            # make MNIST images 32x32x3
            transforms.Grayscale(3),
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,), inplace=True),
        ])

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True,
                               transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000])

        # Assign test dataset or use in dataloader
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform)

        # if stage == "predict" or stage is None:
            #self.mnist_predict = MNIST(self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    # def predict_dataloader(self):
        # return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    # def teardown(self, stage: Optional[str] = None):
        # pass


class PCAMDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32, crop_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(
                                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             transforms.CenterCrop(crop_size)
                                             ])

    def prepare_data(self):
        PCAM(self.data_dir, split="train", download=True)
        PCAM(self.data_dir, split="test", download=True)
        PCAM(self.data_dir, split="val", download=True)

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            self.pcam_train = PCAM(
                self.data_dir, split='train', transform=self.transform)

        if stage == "validate" or stage == "fit" or stage is None:
            self.pcam_val = PCAM(self.data_dir, split='val',
                                 transform=self.transform)

        if stage == "test" or stage is None:
            self.pcam_test = PCAM(
                self.data_dir, split='test', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.pcam_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.pcam_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.pcam_test, batch_size=self.batch_size)

    # def predict_dataloader(self):
        # return DataLoader(self.pcam_predict, batch_size=self.batch_size)

    # def teardown(self, stage: Optional[str] = None):
        # pass
