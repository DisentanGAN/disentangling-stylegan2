"""
Contains PyTorch LightningDataModules for supported datasets.
"""
from typing import Optional

import torch
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST, PCAM

import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.mnist_test = MNIST(self.data_dir, train=False)
        #self.mnist_predict = MNIST(self.data_dir, train=False)
        mnist_full = MNIST(self.data_dir, train=True)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    #def predict_dataloader(self):
        #return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def teardown(self, stage: Optional[str] = None):
        pass

class PCAMDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.pcam_test = PCAM(self.data_dir, split='test')
        self.pcam_train = PCAM(self.data_dir, split='train')
        self.pcam_val = PCAM(self.data_dir, split='val')

    def train_dataloader(self):
        return DataLoader(self.pcam_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.pcam_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.pcam_test, batch_size=self.batch_size)

    #def predict_dataloader(self):
        #return DataLoader(self.pcam_predict, batch_size=self.batch_size)

    def teardown(self, stage: Optional[str] = None):
        pass
