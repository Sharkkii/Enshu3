import numpy as np
import matplotlib.pyplot as plt

# NOTE: neighbor(x, n, r); n is the number of samples, r is the diameter
class HillClimbing:
    def __init__(self, f, neighbor, init_x):
        self.f = f
        self.neighbor = neighbor
        
        self.best_x = init_x
        self.best_fx = f(init_x)
        self.localbest_x = init_x
        self.localbest_fx = f(init_x)
        
        self.report = []
        
    def show(self):
        print("best: f(%.4f) = %.4f" % (self.best_x, self.best_fx))
        print("localbest: f(%.4f) = %.4f" % (self.localbest_x, self.localbest_fx))
    
    def fit(self, limit=100, n=100, r=1.0):
        
        self.best_fx = -np.inf
        self.report = []
        for epoch in range(limit):
            
            self.localbest_fx = -np.inf
            for x in self.neighbor(self.best_x, n=n, r=r):
                fx = self.f(x)
                if (fx > self.localbest_fx):
                    self.localbest_x = x
                    self.localbest_fx = fx
            if (self.best_fx < self.localbest_fx):
                self.best_x = self.localbest_x
                self.best_fx = self.localbest_fx
                self.report.append([self.best_x, self.best_fx])
            else:
                break
        
        print("total epoch: %d" % (epoch+1))
        self.report = np.array(self.report)


# example

# gaussian mixture
def f_example(x):
    dim = np.ndim(x)
    if dim <= 0:
        x = np.array([[x]])
    elif dim <= 1: # batch, NOT features
        x = x[:, None]
    x = x[:, None, :]
    
    # K = 2, D = 1
    pi = np.array([0.6, 0.4])[None, :]
    mu = np.array([2.0, -2.0])[None, :, None]
    sigma = np.array([1.0, 1.0])[None, :]
    
    result = np.exp(- np.sum((x - mu)**2, axis=2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)**(1/2)
    result = pi * result
    result = np.sum(result, axis=1)
    
    if dim <= 0:
        result = result[0]
    
    return result


def neighbor_example(x, n=100, r=1.0):
    return x + np.random.uniform(-r/2,r/2,n)


def main_hillclimbing():

    hc = HillClimbing(f_example, neighbor_example, 1.0)
    hc.fit(limit=100, n=100, r=0.1)
    hc.show()

    x = np.linspace(-10, 10, 1000)
    y = f_example(x)