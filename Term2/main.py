import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.environment import *
from src.helper import *
from src.agent import *

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dest")

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


def train_gail(
    outer_loop = 10,
    inner_loop = 10,
    verbose = True
):

    np.random.seed(0)
    start = time.time()

    env = MDP()
    # env.gamma = 0.2
    dim = 9
    feature_table = np.eye(dim)
    feature_map = lambda state: feature_table[state]
    
    gail = GAIL(env, lam=0.0, minmax=True)
    gail.fit(outer_loop=outer_loop, inner_loop=inner_loop, verbose=verbose)

    end = time.time()
    print(end - start)

    return gail


def train_rl_irl(
    limit_irl = 1000,
    limit_rl = 5000,
    limit_inner_rl = 10,
    verbose = True
):

    np.random.seed(0)
    start = time.time()

    env = MDP()
    # env.gamma = 0.2
    dim = 9
    feature_table = np.eye(dim)
    feature_map = lambda state: feature_table[state]

    rl_irl = RL_IRL(env, feature_map=feature_map, dim=dim)
    rl_irl.fit(limit_irl=limit_irl, limit_inner_rl=limit_inner_rl, limit_rl=limit_rl, verbose=verbose)

    end = time.time()
    print(end - start)

    return rl_irl


def train_max_ent_irl(
    limit_irl = 1000,
    verbose = True
):

    np.random.seed(0)
    start = time.time()

    env = MDP()
    # env.gamma = 0.2
    dim = 9
    feature_table = np.eye(dim)
    feature_map = lambda state: feature_table[state]

    max_ent_irl = Maximum_entropy_IRL(env, feature_map, dim, alpha=1e-2)
    max_ent_irl.fit(limit_irl=limit_irl, verbose=verbose)

    end = time.time()
    print(end - start)

    return max_ent_irl


def eval_gail(gail):

    r = gail.R_template
    d = gail.discriminator

    for s in range(9):
        for a in range(4):
            x = torch.tensor(np.eye(36)[s*4+a], dtype=torch.float32) 
            print("%3e" % d(x).item(), end=" ")
        print()

    for s in range(9):
        for a in range(4):
            print("%3e" % r(d,s,a), end=" ")
        print()
        
    for s in range(9):
        print(gail.policy_agent(s))

    # columns = [(s, a) for s in MDP.STATE_SET for a in MDP.ACTION_SET]
    report_pi = np.array(gail.report_Pi)
    report_r = np.array(gail.report_R)

    cnt = 0
    plt.clf()
    for s in MDP.STATE_SET:    
        for a in MDP.ACTION_SET:
            if MDP.VALID_ACTION_SET[s, a] > 0:
                plt.plot(report_r[:, s, a], label=(s, a), color=colormap_R[cnt])
                cnt += 1
    plt.legend(loc="best")
    plt.savefig(os.path.join(OUTPUT_DIR, "gail_reward.png"))

    plt.clf()
    x = report_pi * MDP.POLICY_OPTIM_TABLE[None]
    x = np.max(x, axis=2)
    for s in MDP.STATE_SET:
        plt.plot(x[:, s], label=s, color=colormap_Pi[s])
    plt.legend(loc="best")
    plt.savefig(os.path.join(OUTPUT_DIR, "gail_optimality.png"))


def eval_rl_irl(rl_irl):

    rl = rl_irl.RL
    irl = rl_irl.IRL

    report_pi = np.array(rl.report_Pi)
    report_r = np.array(irl.report_R)

    plt.clf()
    for s in MDP.STATE_SET:
        plt.plot(report_r[:, s], label=s, color=colormap_R[s])
    plt.legend(loc="best")
    plt.savefig(os.path.join(OUTPUT_DIR, "rl_irl_reward.png"))

    plt.clf()
    x = report_pi * MDP.POLICY_OPTIM_TABLE[None]
    x = np.max(x, axis=2)
    for s in MDP.STATE_SET:
        plt.plot(x[:, s], label=s, color=colormap_Pi[s])
    plt.legend(loc="best")
    plt.savefig(os.path.join(OUTPUT_DIR, "rl_irl_optimality.png"))


def eval_max_ent_irl(max_ent_irl):

    plt.clf()
    report_r = np.array(max_ent_irl.report_R)
    for s in MDP.STATE_SET:
        plt.plot(report_r[:, s], label=s, color=colormap_R[s])
    plt.legend(loc="best")
    plt.savefig(os.path.join(OUTPUT_DIR, "max_ent_irl_reward.png"))


def main():

    # GAIL
    print("#### GAIL ####")
    gail = train_gail(
        outer_loop = 100,
        inner_loop = 10,
        verbose = False
    )
    eval_gail(gail)

    # RL-IRL
    print("#### RL-IRL ####")
    rl_irl = train_rl_irl(
        limit_irl = 100,
        limit_rl = 100,
        limit_inner_rl = 10,
        verbose = False
    )
    eval_rl_irl(rl_irl)

    # Max-Ent-IRL
    print("#### Max-Ent-IRL ####")
    max_ent_irl = train_max_ent_irl(
        limit_irl = 100,
        verbose = False
    )
    eval_max_ent_irl(max_ent_irl)


if __name__ == "__main__":
    main()
