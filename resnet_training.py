
import glob
import os
import time
import csv
import json

import numpy as np
#import matplotlib.pyplot as plt
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torchvision.utils as vutils

from torchvision import transforms
import torchvision.models.resnet as resnet
from torchvision.datasets import MNIST

from torchvision.datasets import PCAM

# from torchsummary import summary

from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm



@torch.no_grad()
def eval_Clf(model, validation_loader):
    model.eval()
    acc = .0
    for i, data in enumerate(validation_loader):
        X = data[0].cuda()
        y = data[1].cuda()
        predicted = torch.argmax(model(X), dim=1)
        # predicted = torch.round(model(X))
        acc+=(predicted == y).sum()/float(predicted.shape[0])
    model.train()
    return (acc/(i+1)).item()



class Training:
    def __init__(self, model, model_name, cuda_device='cuda:0', lr=1e-4, early_stopping_count=20):
        self.model_name = model_name
        self.early_stopping_count = early_stopping_count

        device = 'cpu'
        if torch.cuda.is_available():
            device = cuda_device
        self.device = torch.device(device)

        self.Clf = model.to(self.device)

        self.Clf_opt = optim.Adam(self.Clf.parameters(), lr=lr)
        self.Clf_criterion = torch.nn.CrossEntropyLoss()

        self.path = 'resnet_results/'+model_name

        self.best_acc = .0
        self.best_acc_count = 0

        if not os.path.exists('resnet_results/'):
            os.mkdir('resnet_results/')

        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def train(self, train_loader, validation_loader, epochs):

        self.stats = {
            'clf_loss': [],
            'clf_acc': [],
            'clf_loss_val': [],
            'clf_acc_val': [],
            # todo: recall, precision
        }

        for epoch in tqdm(range(epochs)):
            clf_loss = []
            clf_acc = []


            for i, data in tqdm(enumerate(train_loader)):
                X = data[0].to(self.device)
                y = data[1].to(self.device)

                loss, acc = self._train_Clf(X, y)
                clf_loss.append(loss)
                clf_acc.append(acc)

                

#             if epoch % 10 == 0:
            acc_val = eval_Clf(self.Clf, validation_loader)
            clf_loss_m = sum(clf_loss)/len(clf_loss)
            clf_acc_m = sum(clf_acc)/len(clf_acc)
            self.stats['clf_loss'].append(clf_loss_m)
            self.stats['clf_acc'].append(clf_acc_m)
#             self.stats['clf_loss_val'].append(loss_val)
            self.stats['clf_acc_val'].append(acc_val)

            self.best_acc_count+=1
            if acc_val > self.best_acc:
                self.best_acc = acc_val
                self.best_acc_count = 0
                self.save_best()

            print("Epoch: %d, loss: %f, acc: %.3f, acc_val: %.3f"%(epoch, clf_loss_m, clf_acc_m, acc_val))

            with open(f"resnet_metrics_{self.model_name}.json", "w") as f:
                payload = json.dumps(self.stats)
                f.write(payload)

            if self.best_acc_count >= self.early_stopping_count:
                break

        print("Finished. Best acc: %.3f"%(self.best_acc))
        
        with open(f"resnet_metrics_{self.model_name}.json", "w") as f:
            payload = json.dumps(self.stats)
            f.write(payload)



    def _train_Clf(self, data, labels):
        self.Clf_opt.zero_grad()

        predicted = self.Clf(data)

        loss = self.Clf_criterion(predicted, labels)

        loss.backward()
        self.Clf_opt.step()

        # acc = (np.round(predicted.detach().cpu()) == labels.detach().cpu()).sum()/float(predicted.shape[0])
        acc = (torch.argmax(predicted.detach().cpu(), dim=1) == labels.detach().cpu()).sum()/float(predicted.shape[0])

        return loss.detach().item(), acc.item()

    def save_best(self):
        torch.save(self.Clf.state_dict(), self.path+'/best_model.pth')
        np.save(self.path+'/acc_val.npy', t.stats['clf_acc_val'], allow_pickle=False)
        np.save(self.path+'/acc.npy', t.stats['clf_acc'], allow_pickle=False)
        np.save(self.path+'/loss.npy', t.stats['clf_loss'], allow_pickle=False)


if __name__ == "__main__":
    transform = transforms.Compose(    
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.CenterCrop(32)
            ]    
        )

    loader = DataLoader(
            PCAM('/workspace/ld2/latent_disentanglement/disentangling-stylegan2/data/', split="train", download=False, transform=transform),
            batch_size=32,
            drop_last=True,
        )

    loader_val = DataLoader(
            PCAM('/workspace/ld2/latent_disentanglement/disentangling-stylegan2/data/', split="test", download=False, transform=transform),
            batch_size=32,
            drop_last=True,
            )

    print("initializing training ...")
    t = Training(resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=10), 'clf_resnet18_pcam_train001', early_stopping_count=10)
    print("start training...")
    t.train(loader, loader_val, 100)
