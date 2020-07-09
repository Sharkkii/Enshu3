import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.autograd import Function
import datetime
import time


def timestamp():
    date = datetime.datetime.now()
    year = str(date.year)
    month = str(date.month)
    day = str(date.day)
    hour = str(date.hour)
    minute = str(date.minute)
    second = str(date.second)
    date_string = year + "-" + month + "-" + day + "-" + hour + "-" + minute + "-" + second
    return date_string


def scale_from(x, a, b, eps=1e-4):
# [min, max] -> [-1, 1]
    # return (2*x - a - b) / (b - a + eps) 
# [min, max] -> [0, 1]
    return (x - a) / (b - a + eps)


def max_scale_from(x, eps=1e-4):
    max_scale = np.max(np.abs(x))
    return scale_from(x, - max_scale, max_scale, eps), max_scale


# [0, 1] -> [min, max]
def scale_to(x, a, b):
# [-1, 1] -> [min, max]
    # return (b - a) * x / 2 - (a + b) / 2
# [0, 1] -> [min, max]
    return (b - a) * x + a 


def max_scale_to(x, max_scale):
    return scale_to(x, - max_scale, max_scale)


# NOTE: return max of the designated axis(always return 1-d tensor)
def batch_max(x, max_abs=True):
    if max_abs:
        x = torch.abs(x)
    y = torch.empty(x.size())
    for i in range(x.size(0)):
        y[i] = torch.full(x.size()[1:], torch.max(x[i]))
    return y


def batch_min(x):
    y = torch.empty(x.size())
    for i in range(x.size(0)):
        y[i] = torch.full(x.size()[1:], torch.min(x[i])) 
    return y


def hard_sigmoid(x):
    return clip((x+1)/2, 0, 1)


def hard_tanh(x):
    return clip(x, -1, 1)


# NOTE:
# This function has to do with backpropagation.
# The differential of clip function w.r.t x is 1 if x is inside a section(i.e. not clipped), otherwise 0.
# As long as values are not clipped, the differential is the same as one of non-quantized version. 

class QuantizeFunction(Function):
    @staticmethod
    def forward(context, x, bit, batch):
        if bit is None:
            return x
        scale = batch_max(x, max_abs=True) if batch else torch.max(torch.abs(x))
        x = scale_from(x, - scale, scale)
        x = np.round(x * (2**bit-1)) / (2**bit-1)
        x = scale_to(x, - scale, scale)
        return x
    @staticmethod
    def backward(context, dy):
        return dy, None, None
def quantize(x, bit, batch):
    return QuantizeFunction.apply(x, bit, batch)


# NOTE: clamp function
# class ClipFunction(Function):
#     @staticmethod
#     def forward(context, x, a, b)
#         context.save_for_backward(x)
#         context.a = a
#         context.b = b
#         return np.maximum(a, np.minimum(b, x))
#     @staticmethod
#     def backward(context, dy):
#         x = context.saved_tensor()
#         dx = torch.where(x < a, 0, dy)
#         dx = torch.where(x > b, 0, dx)
#         return dx, None, None
# def clip(x, a, b):
#     return ClipFunction.apply(x, a, b)
        

def unique(x, batch):
    if batch:
        xs = []
        for _x in x:
            _x = _x.detach().numpy()
            xs.append(np.unique(_x))
    else:
        x = x.detach().numpy()
        xs = np.unique(x)
    return xs


# NOTE: (N, C, IH, IW) -> (N*OH*OW, C*K*K)

def image2column(image, kernel_size, stride, padding):
    batch_size, channel, input_height, input_width = image.shape
    output_height = (input_height - kernel_size) // stride + 2 * padding + 1
    output_width = (input_width - kernel_size) // stride + 2 * padding + 1

    image = np.pad(image, [(0,0),(0,0),(padding,padding),(padding,padding)])
    column = np.zeros((batch_size, channel, kernel_size, kernel_size, output_height, output_width))

    for i in range(kernel_size):
        for j in range(kernel_size):
            column[:, :, i, j, :, :] = image[:, :, i:i+output_height, j:j+output_width]
    column = column.transpose((0, 4, 5, 1, 2, 3)).reshape((batch_size * output_height * output_width, -1))

    return column


def column2image(column, input_shape, kernel_size, stride, padding):
    batch_size, channel, input_height, input_width = input_shape
    output_height = (input_height - kernel_size) // stride + 2 * padding + 1
    output_width = (input_width - kernel_size) // stride + 2 * padding + 1

    column = column.reshape((batch_size, output_height, output_width, channel, kernel_size, kernel_size)).transpose((0, 3, 4, 5, 1, 2))
    image = np.zeros((batch_size, channel, input_height + 2 * padding, input_width + 2 * padding))

    # NOTE: for summing up gradients
    for i in range(kernel_size):
        for j in range(kernel_size):
            image[:, :, i:i+output_height, j:j+output_width] += column[:, :, i, j, :, :]
    
    # NOTE: if padding=0, then return 0-shaped tensor
    # image = image[:, :, padding:-padding, padding:-padding]
    image = image[:, :, padding:input_height+padding, padding:input_width+padding]

    return image


def accuracy(y_pred, y_true):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    total_count = y_true.shape[0]
    count = sum(y_true == y_pred)
    return count / total_count
# def accuracy(network, loader):
#     total_count = 0
#     count = 0
#     with torch.no_grad():
#         for i, data in enumerate(loader):
#             x, y = data
#             y_pred = network(x)
#             _, y_pred = torch.max(y_pred, 1)
#             count += sum(y == y_pred).item()
#             total_count += y.size(0)
#     return count / total_count

def search(include, exclude=[], directory="./"):
    files = os.listdir(directory)
    result = ""
    for file in files:
        if all([word in file for word in include]) and not any([word in file for word in exclude]):
            result = file
            break
    return result
