import random
import numpy as np
from function import *

# constants

# RATE = {
#     "reproduce": 0.20,
#     "crossover": 0.60,
#     "mutate": 0.20
# }

# CUMULATIVE_RATE = [
#     RATE["reproduce"],
#     RATE["reproduce"]+RATE["crossover"],
#     RATE["reproduce"]+RATE["crossover"]+RATE["mutate"]
# ]

# NOTE: a Gene class must have genetic operators (and the probabilities to apply them)

# bitstring

class BitStream:
    
    def generate():
        p = BitStream(bits=None)
        return p
    
    def copy(p):
        q = BitStream(bits=p.bits, length=p.bit_length)
        return q
    
    def reproduce(p):
        new_p = BitStream.copy(p)
        return new_p
    
    def crossover(p, q):
        n = random.randrange(1, p.bit_length-1)
        p_former_bits, p_latter_bits = p.cut(n)
        q_former_bits, q_latter_bits = q.cut(n)
        
        new_p_bits = p_former_bits + q_latter_bits
        new_q_bits = q_former_bits + p_latter_bits
        new_p = BitStream(bits=new_p_bits)
        new_q = BitStream(bits=new_q_bits)
        
        return new_p, new_q
        
    def mutate(p):
        n = random.randrange(0, p.bit_length)
        b = "0" if (int(p.bits[n]) > 0) else "1"
        new_p_bit = p.bits[:n] + b + p.bits[n+1:]
        new_p = BitStream(bits=new_p_bit)
            
        return new_p
    
    def genetic_operator(population, probability):
        # NOTE: prob. that which genetic op is applied
        # NOTE: reproduce, crossover, mutate
        Pr = np.array([0.90, 0.95, 1.00])
        # Pr = np.array([0.10, 0.95, 1.00])
        r = np.random.rand()
        
        # Reproduce
        if (r < Pr[0]):
            p = np.random.choice(population, p=probability)
            return BitStream.reproduce(p)
        # CrossOver
        elif (r < Pr[1]):
            p, q = np.random.choice(population, 2, p=probability, replace=False)
            return BitStream.crossover(p, q)
        # Mutate
        else:
            p = np.random.choice(population, p=probability)
            return BitStream.mutate(p)
            
    
    def __init__(self, bits=None, length=32):
        self.bit_length = length
        self.bits = random_bitstream(length) if (bits is None) else bits
        
    def show(self):
        print("%32s" % self.bits)
        
    def cut(self, n):
        former = self.bits[:n]
        latter = self.bits[n:]
        return former, latter


# real value

class RealValue:

    LOWER = -10.0
    UPPER = 10.0
    # NOTE: for BLX-alpha
    # ALPHA = 0.50
    
    def generate():
        p = RealValue(value=None)
        return p
    
    def copy(p):
        q = RealValue(value=p.value)
        return q
    
    def reproduce(p):
        new_p = RealValue.copy(p)
        return new_p
    
    def crossover(p, q, alpha=0.10):
        # BLX-alpha
        new_p_max = max(p.value, q.value) + alpha
        new_p_min = min(p.value, q.value) - alpha
        new_p_value = np.random.uniform(new_p_min, new_p_max)
        if (RealValue.LOWER > new_p_value):
            new_p_value = RealValue.LOWER
        if (RealValue.UPPER < new_p_value):
            new_p_value = RealValue.UPPER
        new_p = RealValue(new_p_value)
        return (new_p,)
        
    def mutate(p, alpha=0.50):
        r = np.random.uniform(-alpha, alpha)
        new_p_value = p.value + r
        if (RealValue.LOWER > new_p_value):
            new_p_value = RealValue.LOWER
        if (RealValue.UPPER < new_p_value):
            new_p_value = RealValue.UPPER
        new_p = RealValue(new_p_value)
        return new_p

    def __init__(self, value=None):
        self.value = random_realvalue(RealValue.LOWER, RealValue.UPPER) if (value is None) else value

    def get(self):
        return self.value

    def set(self, x):
        self.value = x
        
    def show(self):
        print("%.2f" % self.value)
