# import libraries

import numpy as np
import pandas as pd
import time
import datetime

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

from train import *
from qnn import *
from neural_network import *


def main():
    # prepare datasets / dataloaders
    # dataset_train, dataset_test = generate_datasets("mnist")
    dataset_train, dataset_test = generate_datasets("cifar10")
    dataloader_train, dataloader_test = generate_dataloader(dataset_train, dataset_test, batch_train=100, batch_test=1000)
    # dataloader_train, dataloader_test = generate_dataloader(dataset_train, dataset_test, batch_train=100, batch_test=len(dataset_test))

    # define a model
    # 11 parameters
    bit_weight_default = [None, None, None, None, None, None, None, None, None, None, None]
    bit_grad_default = [None, None, None, None, None, None, None, None, None, None, None]
    bit_activation_default = [None, None, None, None, None, None, None, None, None, None]
    bits = {
        "bit_weight": bit_weight_default, 
        "bit_grad": bit_grad_default,
        "bit_activation": bit_activation_default 
    }

    network = QuantizedVgg11(bits)
    optimizer = optim.Adam(network.parameters())
    loss = nn.CrossEntropyLoss()
    trainer = Trainer(network, loss, optimizer)

    # tune
    tuner = Tuner(QuantizedVgg11, dataloader_train, dataloader_test)
    tuner.tune([

        # baseline
        # {"bit_weight": bit_weight_default,
        # "bit_grad": bit_grad_default,
        # "bit_activation": bit_activation_default},

        # [./Csv/first_four_are_quantized_to_2bit_2020-5-11-15-43-20.csv]
        # first4_weight_are_quantized_to_2bit
        # {"bit_weight": [None, 2, 2, 2, 2, None, None, None, None, None, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": bit_activation_default},

        # [./Csv/middle_four_are_quantized_to_2bit_2020-5-11-17-36-35.csv]
        # middle4_weight_are_quantized_to_2bit
        # {"bit_weight": [None, None, None, None, 2, 2, 2, 2, None, None, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": bit_activation_default},

        #  [./Csv/last_three_are_quantized_to_2bit_2020-5-11-18-8-17.csv]
        # last2_weight_are_quantized_to_2bit
        # {"bit_weight": [None, None, None, None, None, None, None, None, 2, 2, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": bit_activation_default},

        # [./Csv/first_middle_seven_are_quantized_to_2bit_2020-5-11-22-26-46.csv]
        # {"bit_weight": [None, 2, 2, 2, 2, 2, 2, 2, None, None, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": bit_activation_default},
        
        # [./Csv/middle_four_weight_and_activation_are_quantized_to_2bit_2020-5-12-0-23-23.csv]
        # middle4_weight_and_activation_are_quantized_to_2bit
        # {"bit_weight": [None, None, None, None, 2, 2, 2, 2, None, None, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": [None, None, None, None, 2, 2, 2, 2, None, None]},

        # middle4_weight_are_quantized_to_8bit
        # {"bit_weight": [None, None, None, None, 8, 8, 8, 8, None, None, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": bit_activation_default},

        # first7_weight_2bit_with_small_lr
        # {"bit_weight": [None, 2, 2, 2, 2, 2, 2, 2, None, None, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": bit_activation_default},
 
        # first3_weight_8bit_and_middle4_weight_2bit_with_small_lr 
        # {"bit_weight": [None, 8, 8, 8, 2, 2, 2, 2, None, None, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": bit_activation_default}
        
        # first4_weight_2bit_with_small_lr_epoch20-40
        # {"bit_weight": [None, 2, 2, 2, 2, None, None, None, None, None, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": bit_activation_default},

        # middle4_weight_2bit_with_small_lr_epoch20-40
        # {"bit_weight": [None, None, None, None, 2, 2, 2, 2, None, None, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": bit_activation_default},
        
        # first7_weight_2bit_with_small_lr
        # {"bit_weight": [None, 2, 2, 2, 2, 2, 2, 2, None, None, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": bit_activation_default},

        # last2_weight_are_quantized_to_2bit
        # {"bit_weight": [None, None, None, None, None, None, None, None, 2, 2, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": bit_activation_default}

        # first4_weight_2bit_with_verysmall_lr_epoch40-60
        # {"bit_weight": [None, 2, 2, 2, 2, None, None, None, None, None, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": bit_activation_default},

        # middle4_weight_2bit_with_verysmall_lr_epoch40-60
        # {"bit_weight": [None, None, None, None, 2, 2, 2, 2, None, None, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": bit_activation_default}

        # first7_weight_2bit_with_verysmall_lr
        # NOTE: why this does not works!
        # {"bit_weight": [None, 2, 2, 2, 2, 2, 2, 2, None, None, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": bit_activation_default},

        # last2_weight_2bit_with_small_lr_epoch20-40
        # {"bit_weight": [None, None, None, None, None, None, None, None, 2, 2, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": bit_activation_default},
        
        # middle4_weight_and_activation_2bit_with_small_lr_epoch20-40
        # {"bit_weight": [None, None, None, None, 2, 2, 2, 2, None, None, None],
        # "bit_grad": bit_grad_default,
        # "bit_activation": [None, None, None, None, 2, 2, 2, 2, None, None]}
        
        # middle4_weight_are_quantized_to_8bit
        {"bit_weight": [None, None, None, None, 8, 8, 8, 8, None, None, None],
        "bit_grad": bit_grad_default,
        "bit_activation": bit_activation_default}

        
    ], [
        # "baseline",
        # "first4_weight_are_quantized_to_2bit",
        # "middle4_weight_are_quantized_to_2bit",
        # "last4_weight_are_quantized_to_2bit"
        # "middle4_weight_and_activation_are_quantized_to_2bit",
        # "middle4_weight_are_quantized_to_8bit"
        # "first7_weight_2bit_with_small_lr"
        # "first3_weight_8bit_and_middle4_weight_2bit_with_small_lr"
        # "first4_weight_2bit_with_small_lr_epoch20-40",
        # "middle4_weight_2bit_with_small_lr_epoch20-40"
        # "first7_weight_2bit_with_small_lr_epoch20-40",
        # "last2_weight_are_quantized_to_2bit_epoch40-60"
        # "first4_weight_2bit_with_verysmall_lr_epoch40-60",
        # "middle4_weight_2bit_with_verysmall_lr_epoch40-60"
        # "first7_weight_2bit_with_verysmall_lr_epoch20-40"
        # "last2_weight_2bit_with_small_lr_epoch20-40",
        # "middle4_weight_and_activation_2bit_with_small_lr_epoch20-40"
        "middle4_weight_8bit_epoch20-40_with_small_lr"

    ], [
       # None,
       # None,
       # None,
       # None,
       # None,
       # None,
       # None,
       # None,
       # "first4_weight_are_quantized_to_2bit_2020-5-12-16-0-51.pth",
       # "middle4_weight_are_quantized_to_2bit_2020-5-12-17-51-21.pth"
       # "first7_weight_2bit_with_small_lr_2020-5-13-10-51-27.pth",
       # None 
       # "first4_weight_2bit_with_small_lr_epoch20-40_2020-5-13-16-10-3.pth",
       # "middle4_weight_2bit_with_small_lr_epoch20-40_2020-5-13-17-59-44.pth"
       # "first7_weight_2bit_with_small_lr_2020-5-13-10-51-27.pth"
       # "last2_weight_are_quantized_to_2bit_new_2020-5-13-22-30-48.pth",
       # "middle4_weight_and_activation_are_quantized_to_2bit_2020-5-12-22-1-49.pth"
       "middle4_weight_are_quantized_to_8bit_2020-5-12-23-51-0.pth"
    ], limit=20)


if __name__ == "__main__":
    main()
    



