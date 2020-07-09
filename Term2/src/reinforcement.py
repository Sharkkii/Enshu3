import numpy as np
from environment import *
from utility import *
from function import *


# Bellman operator

class Bellman:
    def __init__(self, MDP, episode=None):
        self.MDP = MDP
        self.episode = episode
        
    def __call__(self, v):
        trajectory = self.MDP.sample_trajectory(MDP.POLICY_RANDOM)
        numerator = np.zeros((MDP.STATE_N,))
        denominator = np.zeros((MDP.STATE_N,))
        for (state, _, reward, next_state) in self.episode:
            numerator[state] += reward + MDP.DISCOUNT_RATE * v(next_state)
            denominator[state] += 1.0
        expectation = np.where(denominator > 0, numerator/denominator, 0.0)
            
        B_v = lambda state: expectation[state]
        return B_v


# policy iteration method
# TODO: must estimate g, p_T

# class PolicyIteration:
#     def __init__(self, MDP, bellman):
#         self.MDP = MDP
#         self.bellman = bellman
#         self.V = lambda state: 0.0
#         self.pi = lambda state: np.zeros((ACTION_N,))
    
#     def fit(self, limit=100):
#         self.V = lambda state: 0.0
#         for _ in range(limit):
#             self.V = self.bellman(self.V)
#         v = np.array([self.V(state) for state in MDP.STATE_SET)

#         pi = np.eye(MDP.ACTION_N)[np.argmax(v, axis=1)]
#         self.V = lambda state: v[state]
#         self.pi = lambda state: pi[state]


# Bellman operator for action values

class Upsilon:
    def __init__(self, env, episode=None, n=100):
        self.env = env
        # self.episode = episode
        self.episode = []
        for _ in range(n):
            self.episode += env.sample_trajectory(MDP.POLICY_RANDOM, sarsa=True)
        
    def __call__(self, q):
        numerator = np.zeros((MDP.STATE_N, MDP.ACTION_N))
        denominator = np.zeros((MDP.STATE_N, MDP.ACTION_N))
        for (state, action, reward, next_state, next_action) in self.episode:
            numerator[state, action] += reward + MDP.DISCOUNT_RATE * q(next_state, next_action)
            denominator[state, action] += 1.0
        expectation = np.where(denominator > 0, numerator/denominator, 0.0)

        Upsilon_q = lambda state, action: expectation[state, action]
        return Upsilon_q


# value iteration method

class ValueIteration:
    def __init__(self, env, upsilon):
        self.env = env
        self.upsilon = upsilon
        self.Q = lambda state, action: 0.0
        self.V = lambda state: 0.0
        self.pi = lambda state: np.zeros((MDP.ACTION_N,))
    
    def fit(self, limit=100):
        self.Q = lambda state, action: 0.0
        for _ in range(limit):
            self.Q = self.upsilon(self.Q)
        q = np.array([self.Q(state, action) for state in MDP.STATE_SET for action in MDP.ACTION_SET]).reshape((MDP.STATE_N, MDP.ACTION_N))
        q = np.where(MDP.VALID_ACTION_SET > 0.0, q, -np.inf)
        # print(q)

        v = np.max(q, axis=1)
        pi = np.eye(MDP.ACTION_N)[np.argmax(q, axis=1)]
        self.V = lambda state: v[state]
        self.pi = lambda state: pi[state]


# policy approximator (parametrize each entry of STATE * ACTION table)

class Pi_theta:
    
    def __init__(self, env):
        self.env = env
        self.weight = np.zeros((MDP.STATE_N, MDP.ACTION_N))
        
    def __call__(self, state):
        score = self.weight[state]
        score = np.where(MDP.VALID_ACTION_SET[state] > 0, score, -np.inf)
        distribution = softmax(score)
        return distribution
    
    def log_gradient(self, state, action):
        grad = np.zeros((MDP.STATE_N, MDP.ACTION_N))
        # DEBUG:
        # grad[state] = - self(state)
        # grad[state, action] += 1.0
        for s in MDP.STATE_SET:
            for a in MDP.ACTION_SET:
                if s == state:
                    if a == action:
                        grad[s, a] += 1.0
                    grad[s, a] -= self(s)[a]
        return grad


# Policy Gradient Method
# NOTE: the optimized policy must have `weight` property and `log_gradient` method
# learning rate has a large influence on "success"!

class PolicyGradient:
    
    def __init__(self, env, policy, lam=1e-2):
        self.env = env
        self.policy = policy
        self.alpha = lambda t: 1/((t+1)*100)
        self.lam = lam
        self.entropy = lambda s, a: - np.log(self.policy(s)[a])

        self.report_R = []
        self.report_Pi = []

    # H(pi) = E[- log pi(a|s)]
    # def causal_entropy(self):
    # def grad_entropy(self):
    
    # caution: large c may lead to fallacy
    def update(self, episode, descent=False):
        sign = -1 if descent else 1
        episode_H = []
        for sars in episode:
            s, a, _, ns = sars
            r_H = self.entropy(s, a)
            episode_H.append((s, a, r_H, ns))

        b = np.mean(np.array(episode)[:, 2])
        b_H = np.mean(np.array(episode_H)[:, 2])

        for t, sars in enumerate(episode):
            s, a, r, _ = sars
            c = c_return(self.env, episode, t)
            c_H = c_return(self.env, episode_H, t)
            print("(%d,%d)" % (s,a))
            print("c", c)
            print("b", b)
            print("c-b", c-b)
            self.policy.weight = self.policy.weight + sign * self.alpha(t) * self.policy.log_gradient(s, a) * ((c - b) - self.lam * (c_H - b_H))

        self.report_Pi.append([self.policy(s) for s in MDP.STATE_SET])
        

    def fit(self, limit=100, verbose=False):
        for i in range(limit):
            if verbose:
                print("\rRL % d" % i, end="")
            episode = self.env.sample_trajectory(self.policy)
            self.update(episode)
        if verbose:
            print("")