# NOTE: PyTorch
# import libraries

import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, Dataset, Subset
from function import *


# MNIST dataset

mnist_classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0")


# CIFAR10 dataset

cifar10_classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


# transform, datasets, dataloader

def generate_datasets(name):
    if (name == "mnist"):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
        dataset_train = MNIST(root="./Mydatasets", train=True, download=False, transform=transform)
        dataset_test = MNIST(root="./Mydatasets", train=False, download=False, transform=transform)
    elif (name == "cifar10"):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        dataset_train = CIFAR10(root="./Mydatasets", train=True, download=False, transform=transform)
        dataset_test = CIFAR10(root="./Mydatasets", train=False, download=False, transform=transform)
    else:
        transform = None
        dataset_train = None
        dataset_test = None
    return dataset_train, dataset_test


def generate_dataloader(dataset_train, dataset_test, batch_train=1, batch_test=1):
    dataloader_train = DataLoader(dataset_train, batch_size=batch_train, shuffle=True, num_workers=2)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_test, shuffle=False, num_workers=2)
    return dataloader_train, dataloader_test


# TODO:

class Trainer:
    def __init__(self, net, loss, optim):
        self.net = net
        self.optim = optim
        self.loss = loss
        self.scoring_report = None
        # self.report = None
    
    def fit(self, loader_train, loader_test=None, limit=1):
        # print(self.net)
        print("start")
        start = time.time()

        try:

            score_buffer = []

            for epoch in range(limit):
                total_loss = 0.0
                interval = 10
                buffer = []

                # TODO: rewrite so as not to calculate the same forwarding process twice
                # if possible, separate loss-layer apart from other layers
                for idx, data in enumerate(loader_train):
                    x_train, y_train = data
                    self.optim.zero_grad()
                    y_train_pred = self.net(x_train)
                    loss = self.loss(y_train_pred, y_train)
                    total_loss += loss.item()
                    loss.backward()
                    # NOTE: do something
                    if hasattr(self.net, "f"):
                        self.net.f()
                    self.optim.step()
                
                    buffer.append((y_train_pred.detach().numpy(), y_train.detach().numpy()))

                    if (idx % interval == interval-1):
                        total_loss /= interval

                        y_train_preds, y_trains = zip(*buffer)
                        y_train_preds = np.concatenate(y_train_preds)
                        y_trains = np.concatenate(y_trains)
                        # NOTE: one hot encoding
                        y_trains = np.eye(10)[y_trains]
                        accuracy_train = accuracy(y_train_preds, y_trains)

                        if (loader_test):
                            x_test, y_test = iter(loader_test).next()
                            y_test_pred = self.net(x_test).detach().numpy()
                            # NOTE: one hot encoding
                            y_test = np.eye(10)[y_test.detach().numpy()]
                            accuracy_test = accuracy(y_test_pred, y_test)
                            # NOTE: for score report
                            checkpoint = time.time()
                            score_buffer.append([epoch, idx//interval, checkpoint - start, total_loss, accuracy_train, accuracy_test])
                            print("epoch %d, iter %d, time: %.2f" % (epoch, idx//interval, checkpoint - start))
                            print("loss: %5.3f, accuracy(train): %.3f, accuracy(test): %.3f" % (total_loss, accuracy_train, accuracy_test))
                            
                        else:
                            # NOTE: for score report
                            checkpoint = time.time()
                            score_buffer.append([epoch, idx//interval, checkpoint - start, total_loss, accuracy_train])
                            print("epoch %d, iter %d, time: %.2f" % (epoch, idx//interval, checkpoint - start))
                            print("loss: %5.3f, accuracy(train): %.3f" % (total_loss, accuracy_train))

                        total_loss = 0
                        buffer = []
        except KeyboardInterrupt:
            pass
        finally:
            # NOTE: for score report
            column_names = ["epoch", "iter", "time", "loss", "accuracy(train)"]
            if loader_test:
                column_names.append("accuracy(test)")
            self.score_report = pd.DataFrame(score_buffer, columns=column_names)

        end = time.time()
        print("finish")

    def export(self, content, filename, directory="./"):
        stamp = timestamp()
        csv_path = directory+"Csv/"+filename+"_"+stamp+".csv"
        model_path = directory+"Mymodels/"+filename+"_"+stamp+".pth"
        torch.save(self.net.state_dict(), model_path)
        self.score_report.to_csv(csv_path, index=False)
        report = open(directory+"Csv/"+"report.txt", "a")
        report.write("["+csv_path+"]")
        report.write("\n\n")
        report.write(content)
        report.write("\n\n\n")
        report.close()

        print("export the score report!")

# Tuner

class Tuner:
    def __init__(self, template, dataloader_train, dataloader_test=None):
        self.template = template
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test

    def tune(self, hyperparams, names, load_state_paths, limit=1):
        for (hyperparam, name, load_state_path) in zip(hyperparams, names, load_state_paths):
            net = self.template(hyperparam)
            if load_state_path:
                net.load_state_dict(torch.load("./Mymodels/"+load_state_path))
                print("load parameters from " + load_state_path)
            loss = nn.CrossEntropyLoss()
            # NOTE: small: lr=0.0001
            # NOTE: verysmall: lr=0.00001
            optimizer = optim.Adam(net.parameters(), lr=0.0001)
            trainer = Trainer(net, loss, optimizer)
            print(hyperparam)
            trainer.fit(self.dataloader_train, self.dataloader_test, limit=limit)

            content = str(hyperparam)
            if load_state_path:
                content += "\nuse pre-trained parameters of " + load_state_path
            trainer.export(content=content, filename=name)
