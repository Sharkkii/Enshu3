# import libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from qnn import *

# VGG11 network

class Vgg11(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.block0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block5 = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )
        
    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.block5(x)
        return x

# param = { "bit_weight": [], "bit_grad": [], "bit_act": [] }

class QuantizedVgg11(torch.nn.Module):
    
    def __init__(self, bits):
        super().__init__()
        self.bit_weight = bits["bit_weight"]
        self.bit_grad = bits["bit_grad"]
        self.bit_act = bits["bit_activation"]
        self.convolution_layers = [
            QuantizedConv2d(3, 64, kernel_size=3, padding=1, bit_data=self.bit_weight[0], bit_grad=self.bit_grad[0]),
            QuantizedConv2d(64, 128, kernel_size=3, padding=1, bit_data=self.bit_weight[1], bit_grad=self.bit_grad[1]),
            QuantizedConv2d(128, 256, kernel_size=3, padding=1, bit_data=self.bit_weight[2], bit_grad=self.bit_grad[2]),
            QuantizedConv2d(256, 256, kernel_size=3, padding=1, bit_data=self.bit_weight[3], bit_grad=self.bit_grad[3]),
            QuantizedConv2d(256, 512, kernel_size=3, padding=1, bit_data=self.bit_weight[4], bit_grad=self.bit_grad[4]),
            QuantizedConv2d(512, 512, kernel_size=3, padding=1, bit_data=self.bit_weight[5], bit_grad=self.bit_grad[5]),
            QuantizedConv2d(512, 512, kernel_size=3, padding=1, bit_data=self.bit_weight[6], bit_grad=self.bit_grad[6]),
            QuantizedConv2d(512, 512, kernel_size=3, padding=1, bit_data=self.bit_weight[7], bit_grad=self.bit_grad[7]),
        ]
        self.linear_layers = [
            QuantizedLinear(512, 4096, bit_data=self.bit_weight[8], bit_grad=self.bit_grad[8]),
            QuantizedLinear(4096, 4096, bit_data=self.bit_weight[9], bit_grad=self.bit_grad[9]),
            QuantizedLinear(4096, 10, bit_data=self.bit_weight[10], bit_grad=self.bit_grad[10]),
        ]
        self.activation_layers = [
            QuantizedReLU(bit=self.bit_act[0], dynamic_range=True),
            QuantizedReLU(bit=self.bit_act[1], dynamic_range=True),
            QuantizedReLU(bit=self.bit_act[2], dynamic_range=True),
            QuantizedReLU(bit=self.bit_act[3], dynamic_range=True),
            QuantizedReLU(bit=self.bit_act[4], dynamic_range=True),
            QuantizedReLU(bit=self.bit_act[5], dynamic_range=True),
            QuantizedReLU(bit=self.bit_act[6], dynamic_range=True),
            QuantizedReLU(bit=self.bit_act[7], dynamic_range=True),
            QuantizedReLU(bit=self.bit_act[8], dynamic_range=True),
            QuantizedReLU(bit=self.bit_act[9], dynamic_range=True)
        ]

        self.block0 = nn.Sequential(
            self.convolution_layers[0],
            nn.BatchNorm2d(64),
            self.activation_layers[0],
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block1 = nn.Sequential(
            self.convolution_layers[1],
            nn.BatchNorm2d(128),
            self.activation_layers[1],
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            self.convolution_layers[2],
            nn.BatchNorm2d(256),
            self.activation_layers[2],
            self.convolution_layers[3],
            nn.BatchNorm2d(256),
            self.activation_layers[3],
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            self.convolution_layers[4],
            nn.BatchNorm2d(512),
            self.activation_layers[4],
            self.convolution_layers[5],
            nn.BatchNorm2d(512),
            self.activation_layers[5],
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            self.convolution_layers[6],
            nn.BatchNorm2d(512),
            self.activation_layers[6],
            self.convolution_layers[7],
            nn.BatchNorm2d(512),
            self.activation_layers[7],
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block5 = nn.Sequential(
            self.linear_layers[0],
            nn.Dropout(0.5),
            self.linear_layers[1],
            nn.Dropout(0.5),
            self.linear_layers[2]
            # nn.Linear(4096, 10)
        )
        
    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.block5(x)
        return x


# FIXME:
class Hoge(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            QuantizedConv2d(1,3,3,1,1,bit=1),
            QuantizedConv2d(3,3,3,1,1,bit=1),
            QuantizedConv2d(3,1,3,1,1,bit=1)
        ]
        self.a = nn.Sequential(
            self.layers[0],
            # nn.Conv2d(1,3,3,1,1),
            QuantizedReLU(bit=1),
            # nn.ReLU(),
            self.layers[1],
            # nn.Conv2d(3,3,3,1,1),
            # nn.ReLU(),
            QuantizedReLU(bit=1),
            self.layers[2],
            # nn.Conv2d(3,1,3,1,1),
            # nn.ReLU()
            QuantizedReLU(bit=1)
        )
        self.b = QuantizedLinear(28*28, 10, bit=1)
        # self.b = nn.Linear(28*28, 10)
    def forward(self, x):
        x = self.a(x)
        x = x.view(x.size(0), -1)
        x = self.b(x)
        return x

class Hoge(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            QuantizedConv2d(1,3,3,1,1,bit_data=2, bit_grad=2),
            QuantizedConv2d(3,3,3,1,1,bit_data=2, bit_grad=2),
            QuantizedConv2d(3,1,3,1,1,bit_data=2, bit_grad=2)
        ]
        self.relus = [
            QuantizedReLU(bit=2, dynamic_range=True),
            QuantizedReLU(bit=2, dynamic_range=True),
            QuantizedReLU(bit=2, dynamic_range=True)
        ]
        self.out = QuantizedLinear(28*28, 10, bit_data=8, bit_grad=None)
        # self.out = nn.Linear(28*28, 10)
    def f(self):
        for layer in self.layers:
            layer.f()
        for layer in self.relus:
            layer.f()
    def forward(self, x):
        q = Quantizer(bit_data=None, bit_grad=1)
        x = self.layers[0](x)
        # time.sleep(1.0)
        # print(x[0])
        x = q(x)
        x = self.relus[0](x)
        # time.sleep(1.0)
        # print(x[0])
        x = self.layers[1](x)
        # time.sleep(1.0)
        # print(x[0])
        x = q(x)
        x = self.relus[1](x)
        # time.sleep(1.0)
        # print(x[0])
        x = self.layers[2](x)
        # time.sleep(1.0)
        # print(x[0])
        x = q(x)
        x = self.relus[2](x)
        # time.sleep(1.0)
        # print(x[0])
        x = x.view(x.size(0), -1)
        # time.sleep(1.0)
        # print(x[0])
        x = self.out(x)
        return x