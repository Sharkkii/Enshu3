import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.distributions.uniform import Uniform
from helper import *

class MyTemplate(nn.Module):
    def __init__(self, hyperparam):
        super().__init__()
        a, b, c = hyperparam
        self.layer0 = nn.Sequential(
            QuantizedConv2d(1,3,3,1,bit_data=a),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            QuantizedConv2d(3,1,3,1,bit_data=b),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            QuantizedLinear(24*24,100,bit_data=c),
            nn.BatchNorm1d(100),
            # QuantizedLinear(100,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )
    def forward(self, x):
        x = self.layer0(x)
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        return x

class QuantizedLinear(nn.Module):
    def __init__(self, input_feature, output_feature, bit_data=2, bit_grad=None, keep_w=True):
        super().__init__()
        self.w = None
        self.linear = nn.Linear(input_feature, output_feature)
        self.bit_data = bit_data
        self.bit_grad = bit_grad
        self.keep_w = keep_w
    def f(self):
        if self.keep_w:
            self.linear.weight.data = self.w
        if self.bit_grad:
            self.linear.weight.grad = quantize(self.linear.weight.grad, bit=self.bit_grad, batch=False)
    def forward(self, x):
        w = self.linear.weight.data
        self.w = w
        x_quantized = quantize(x, bit=self.bit_data, batch=True)
        w_quantized = quantize(w, bit=self.bit_data, batch=False)
        self.linear.weight.data = w_quantized
        y = self.linear(x_quantized)
        return y

class QuantizedConv2d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bit_data=2, bit_grad=None, keep_w=True):
        super().__init__()
        self.w = None
        self.conv2d = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding)
        self.bit_data = bit_data
        self.bit_grad = bit_grad
        self.keep_w = keep_w
    def f(self):
        if self.keep_w:
            self.conv2d.weight.data = self.w
        if self.bit_grad:
            self.conv2d.weight.grad = quantize(self.conv2d.weight.grad, bit=self.bit_grad, batch=False)

    def forward(self, x):
        w = self.conv2d.weight.data
        self.w = w
        x_quantized = quantize(x, bit=self.bit_data, batch=True)
        w_quantized = quantize(w, bit=self.bit_data, batch=False)
        self.conv2d.weight.data = w_quantized
        y = self.conv2d(x_quantized)
        return y

class QuantizedReLU(nn.Module):
    def __init__(self, bit, dynamic_range=False, lr=0.001):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.bit = bit
        self.y = None
        self.y_quantized = None
        self.dynamic_range = dynamic_range
        self.lr = lr
    def f(self):
        if self.dynamic_range:
            self.alpha.grad = torch.sum(torch.mean(self.y_quantized - self.y, axis=0))
            self.alpha.data -= self.lr * self.alpha.grad
    def forward(self, input_x):
        output_y = torch.clamp(input_x, 0, self.alpha.data)
        self.y = output_y
        output_y_quantized = quantize(output_y, bit=self.bit, batch=True)
        self.y_quantized = output_y_quantized
        return output_y_quantized

class QuantizerFunction(Function):
    @staticmethod
    def forward(context, x, bit_data=None, bit_grad=None):
        context.bit_data = bit_data
        context.bit_grad = bit_grad
        x_quantized = quantize(x, bit=bit_data, batch=True)
        return x_quantized
    @staticmethod
    def backward(context, dy):
        bit_data = context.bit_data
        bit_grad = context.bit_grad
        dy_quantized = quantize(dy, bit=bit_grad, batch=True)
        return dy, None, None

class Quantizer(nn.Module):
    def __init__(self, bit_data=None, bit_grad=None):
        super().__init__()
        self.bit_data = bit_data
        self.bit_grad = bit_grad
    def forward(self, x):
        return QuantizerFunction.apply(x, self.bit_data, self.bit_grad)
        