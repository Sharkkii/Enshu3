import numpy as np
import matplotlib.pyplot as plt
from environment import *


# convert function into the corresponding vector, and vice versa
# TODO: generalize

def f2v(f, D):
    result = np.empty(D)
    for d in range(D[0]):
        result[d] = f(d)
    return result

def v2f(v):
    result = lambda d: v[d]
    return result


# extract s/sa-trajectory from given sars-trajectory

def sars2s(trajectory):
    sars = np.array(trajectory, dtype=int)
    s = sars[:, 0]
    s = list(s)
    return s

def sars2sa(MDP, sars):
    sars = np.array(sars, dtype=int)
    s, a = sars[:, 0], sars[:, 1]
    sa = s * MDP.ACTION_N + a
    sa = np.eye(MDP.STATE_N * MDP.ACTION_N)[sa]
    return sa


# TODO: clean this room

def softmax(x, beta=1.0):
    score = np.exp(beta * x)
    distribution = score / np.sum(score)
    return distribution


# plot function

def reportR(filename, title, save=None):
    report_r = pd.read_csv(filename)
    report_r = report_r.values.reshape(-1, 9, 4)
    
    colormap_R = [
        [1.0, 0.0, 0.0, 1.0], # (0,1)* Red
        [1.0, 0.0, 0.0, 0.2], # (0,3)  
        [1.0, 0.5, 0.0, 1.0], # (1,2)* Orange
        [1.0, 0.5, 0.0, 0.2], # (1,3)
        [1.0, 1.0, 0.0, 0.2], # (2,1)
        [1.0, 1.0, 0.0, 1.0], # (2,2)* Yellow
        [0.5, 1.0, 0.0, 0.2], # (3,0)  
        [0.5, 1.0, 0.0, 0.2], # (3,1)
        [0.5, 1.0, 0.0, 1.0], # (3,3)* Light Green
        [0.0, 1.0, 0.0, 1.0], # (4,1)* Green
        [0.0, 1.0, 0.0, 0.2], # (4,2)
        [0.0, 0.5, 0.5, 1.0], # (5,0)* SkyBlue
        [0.0, 0.0, 1.0, 1.0], # (6,0)* Blue
        [0.5, 0.0, 1.0, 0.2], # (7,0)
        [0.5, 0.0, 1.0, 1.0], # (7,3)* Purple
        [1.0, 0.0, 1.0, 1.0], # (8,2)* Magenta
    ]

    cnt = 0
    for s in MDP.STATE_SET:    
        for a in MDP.ACTION_SET:
            if MDP.VALID_ACTION_SET[s, a] > 0:
                plt.plot(report_r[:, s, a], label=(s, a), color=colormap_R[cnt])
                cnt += 1
    plt.legend()
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("reward")
    
    if save:
        plt.savefig(save)
        
    plt.show()
    

def reportPi(filename, title, save=None):
    report_pi = pd.read_csv(filename)
    report_pi = report_pi.values.reshape(-1, 9, 4)
    
    colormap_Pi = [
        [1.0, 0.0, 0.0, 1.0], # (0,1)* Red
        [1.0, 0.5, 0.0, 1.0], # (1,2)* Orange
        [1.0, 1.0, 0.0, 1.0], # (2,2)* Yellow
        [0.5, 1.0, 0.0, 1.0], # (3,3)* Light Green
        [0.0, 1.0, 0.0, 1.0], # (4,1)* Green
        [0.0, 0.5, 0.5, 1.0], # (5,0)* SkyBlue
        [0.0, 0.0, 1.0, 1.0], # (6,0)* Blue
        [0.5, 0.0, 1.0, 1.0], # (7,3)* Purple
        [1.0, 0.0, 1.0, 1.0], # (8,2)* Magenta
    ]
    
    x = report_pi * MDP.POLICY_OPTIM_TABLE[None]
    x = np.max(x, axis=2)
    for s in MDP.STATE_SET:
        plt.plot(x[:, s], label=s, color=colormap_Pi[s])
    plt.legend()
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("probability")
    
    if save:
        plt.savefig(save)
    
    plt.show()
