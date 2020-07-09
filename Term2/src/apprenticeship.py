import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from environment import *
from utility import *
from function import *
from reinforcement import *
from inverse_reinforcement import *


# naive RL-IRL

class RL_IRL:
    def __init__(self, env, feature_map, dim):
        self.env = env
        self.IRL = Maximum_entropy_IRL(self.env, feature_map, dim)
        self.RL = PolicyGradient(self.env, Pi_theta(self.env))

    def fit(self, limit_irl=10, limit_inner_rl=10, limit_rl=10, verbose=False):
        self.IRL.fit(limit_irl, limit_inner_rl, verbose=verbose)
        self.RL.fit(limit_rl, verbose=verbose)


# Discriminator, which is a part of GAIL

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(36, 1000)
        self.hidden_layer = nn.Linear(1000, 1000)
        self.next_hidden_layer = nn.Linear(1000, 1000)
        self.output_layer = nn.Linear(1000, 1)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = nn.ReLU()(x)
        x = self.hidden_layer(x)
        x = nn.ReLU()(x)
        # x = self.next_hidden_layer(x)
        # x = nn.ReLU()(x)
        x = self.output_layer(x)
        x = nn.Sigmoid()(x)
        return x


# Generative Adversarial Imitation Learning; GAIL
# NOTE: minimize E - lam H (with R = log(1 - D)): default
# NOTE: maximize E + lam H (with R = log D)
# - log(1 - D) -> log D
# when minimizing, lambda must be positive
# when maximizing, lambda must be negative

class GAIL:
    def __init__(self, env, lam=0.0, minmax=False):
        self.env = env
        self.minmax = minmax
        self.lam = - lam if self.minmax else lam
        self.policy_expert = MDP.POLICY_EXPERT
        self.policy_agent = Pi_theta(self.env)
        self.policy_gradient = PolicyGradient(env, self.policy_agent, self.lam)
        self.discriminator = Discriminator()
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-3)

        self.report_R = []
        self.report_Pi = []
        
    def R_template(self, discriminator, s, a, eps=1e-8):
        x = np.eye(MDP.STATE_N * MDP.ACTION_N)
        x = x[int(s * MDP.ACTION_N + a)]
        x = torch.tensor(x, dtype=torch.float32)
        x = discriminator(x).item()
        if self.minmax:
            # NOTE: minimize log(1 - D(s,a))
            x = np.log(1 - x + eps)
        else:
            # NOTE: maximize log(D(s,a))
            x = np.log(x + eps)
        return x
        
    def fit(self, outer_loop, inner_loop, eps=1e-8, verbose=False):
        trajectory_sars_expert = self.env.sample_trajectory(self.policy_expert)
        trajectory_sa_expert = sars2sa(self.env, trajectory_sars_expert)
        trajectory_sa_expert = torch.tensor(trajectory_sa_expert, dtype=torch.float32)

        for i in range(outer_loop):
            
            # IRL step: [MAXMIZE] optimize w of Discriminator
            
            if verbose:
                print("\rIRL %d" % i, end="")

            for j in range(inner_loop):
                
                self.optimizer.zero_grad()

                trajectory_sars_agent = self.env.sample_trajectory(self.policy_agent)
                trajectory_sa_agent = sars2sa(self.env, trajectory_sars_agent)
                trajectory_sa_agent = torch.tensor(trajectory_sa_agent, dtype=torch.float32)

                loss = torch.mean(torch.log(self.discriminator(trajectory_sa_expert) + eps)) + torch.mean(torch.log(1 - self.discriminator(trajectory_sa_agent) + eps))
                # instead of maximization, we minimize the negative
                loss = loss * (-1)
                loss.backward()
                
                self.optimizer.step()
        
            # RL step: [MINIMIZE] optimize theta of policy
            
            if verbose:
                print("\rRL %d " % i, end="")

            r = lambda s, a: self.R_template(self.discriminator, s, a)
            self.env.update_R(r)

            if self.minmax:
                # NOTE: minimize log(1 - D(s,a))
                self.policy_gradient.update(trajectory_sars_agent, descent=True)
            else:
                # NOTE: maximize log(D(s,a))
                self.policy_gradient.update(trajectory_sars_agent, descent=False)


            self.report_R.append(np.array([[r(s, a) for a in MDP.ACTION_SET] for s in MDP.STATE_SET]))
            # self.report_Pi.append(self.policy_gradient.report_Pi[i])
        
        self.report_Pi = self.policy_gradient.report_Pi
        
        if verbose:
            print("")
