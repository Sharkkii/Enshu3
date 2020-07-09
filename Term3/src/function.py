import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import struct


def top_n(x, n=1, desc=True, return_indices=False):
    k, v = zip(*x)
    i = np.argsort(v)
    if desc:
        i = i[::-1]
    x = [(k[i[j]],v[i[j]]) for j in range(0,n)]
    if return_indices:
        return x, i[:n]
    else:
        return x

def augment_last(x, n=1):
    x_augmented = np.empty((n, *x.shape[1:]))
    x_augmented[:] = x[-1]
    x = np.concatenate([x, x_augmented], axis=0)
    return x


def random_bitstream(length):
    bit_string = ""
    for _ in range(length):
        r = np.random.rand()
        if (r > 0.50):
            bit_string += "1"
        else:
            bit_string += "0"
    return bit_string


def random_realvalue(a, b):
    value = np.random.uniform(a, b)
    return value

        
def scale_from(x, a, b, eps=1e-4):
    return (2*x - a - b) / (b - a + eps)


def scale_to(x, a, b):
    return (b - a) * x / 2 - (a + b) / 2


def integer_to_float(x, lower_bound, upper_bound):
    b = len(x)
    
    y = 0
    for i in range(b):
        y += int(x[i]) * 2**(b-1-i)
    y -= int(x[0]) * 2**b
    
    y = scale_from(y, -2**(b-1), 2**(b-1))
    y = np.round(y * (2**b-1)) / (2**b-1)
    y = scale_to(y, lower_bound, upper_bound)
    
    return y


def softmax(x):
    y = np.exp(x)
    y = y / np.sum(y)
    return y


def byte_float32_to_binary_int(x):
    x = struct.unpack(">I", x)
    x = bin(x[0])
    # strip "0b"
    x = x[2:].zfill(32)
    return x


def binary_int_to_byte_float32(x):
    x = int(x)
    x = struct.pack(">f", x)
    return x


def binary_int_to_float(x):
    x = int(x)
    x = struct.pack(">f", x)
    x = struct.unpack(">f", x)
    return x
    