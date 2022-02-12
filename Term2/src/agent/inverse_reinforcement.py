import numpy as np
from ..environment import *
from ..helper import *
from .reinforcement import *


# reward approximator (linear with some feature mapping)

class R_theta:
    def __init__(self, feature_map, dim):
        self.dim = dim
        # self.weight = np.random.randn(dim)
        self.weight = np.zeros((dim,))
        self.feature_map = feature_map
            
    def __call__(self, state, dummy_action=None, is_trajectory=False):
        if is_trajectory:
            r = 0.0
            for s in state:
                r += self(s, None, is_trajectory=False)
        else:
            r = np.dot(self.feature_map(state), self.weight)
        return r    
    
    def grad(self, state, dummy_action=None, is_trajectory=False):
        if is_trajectory:
            g = 0.0
            for s in state:
                g += self(s, None, is_trajectory=False)
        else:
            g = self.feature_map(state)
        return g
    

# Maximum Entropy Inverse Reinforcement Learning
# NOTE: the optimized reward function must have `weight` property and `grad` method

class Maximum_entropy_IRL:
    def __init__(self, env, feature_map, dim, alpha=1e-2):
        self.env = env
        self.feature_map = feature_map
        self.dim = dim
        self.R_approx = R_theta(feature_map, dim)
        self.policy_expert = MDP.POLICY_EXPERT
        self.alpha = alpha
        
        # NOTE:
        self.report_R = []
        self.report_Pi = []

        self.env.update_R(self.R_approx)
    
    # FIXME: return table...
    def pi_approx(self, limit_rl=10, T=10):

        # NOTE: use RL(value iteration method)
        if limit_rl > 0:

            upsilon = Upsilon(self.env, episode=None)
            value_iteration = ValueIteration(self.env, upsilon)
            value_iteration.fit(limit=limit_rl)
            pi = value_iteration.pi
            pi = np.array([pi(state) for state in MDP.STATE_SET])

        # NOTE: calculate partition function Z
        else:
        
            Z_s = np.ones((MDP.STATE_N,))
            Z_sa = np.empty((MDP.STATE_N, MDP.ACTION_N))
            
            r = np.array([self.R_approx(s) for s in MDP.STATE_SET])
            p_T = MDP.TRANSITION_TABLE
            
            for _ in range(T):
                Z_sa = np.zeros_like(Z_sa)
                for s in MDP.STATE_SET:
                    for a in MDP.ACTION_SET:
                        for s_prime in MDP.STATE_SET:
                            Z_sa[s, a] +=  np.exp(r[s]) * p_T[s, a, s_prime] * Z_s[s_prime]
                Z_s = np.zeros_like(Z_s)
                for s in MDP.STATE_SET:
                    for a in MDP.ACTION_SET:
                        Z_s[s] += Z_sa[s, a]
        
            Z_s = Z_s.reshape(-1, 1)
            pi = Z_sa / Z_s
        
        return pi
                        
        
    def expected_state_visitation_frequency(self, pi, T=10):
        
        p_T = MDP.TRANSITION_TABLE
        # Ds = np.repeat(f2v(MDP.INITIAL_DISTRIBUTION, (MDP.STATE_N,1)), T, axis=1)
        Ds = np.zeros((MDP.STATE_N, T))
        Ds[0, 0] = 1.0
        # Ds[:, 0] = f2v(MDP.INITIAL_DISTRIBUTION, (MDP.STATE_N,))
        for t in range(1, T):
            for s in MDP.STATE_SET:
                for a in MDP.ACTION_SET:
                    for s_prime in MDP.STATE_SET:
                        Ds[s, t] += pi[s_prime, a] * p_T[s_prime, a, s] * Ds[s_prime, t-1]
        Ds = np.array(Ds)
        D = np.sum(Ds, axis=1)
        return D

    # feature count of expert
    def empirical_feature_count(self, episodes_expert):
        F = np.zeros((self.dim,))
        for episode_expert in episodes_expert:
            for state, _, _, _ in episode_expert:
                F += self.feature_map(state)
        F /= len(episodes_expert)
        return F

    # feature count of agent
    def expected_feature_count(self, policy_agent, episodes_expert):
        F = np.zeros((self.dim,))
        T = int(np.mean([len(episode) for episode in episodes_expert]))
        D = self.expected_state_visitation_frequency(policy_agent, T=T)
        # for episode_expert in episodes_expert:
        #     for state, _, _, _ in episode_expert:
        #         F += D[state] * self.feature_map(state)
        for state in MDP.STATE_SET:
            F += D[state] * self.feature_map(state)
        return F
    
    
    # TODO: is episode really "expert trajectory"?
    def update(self, episodes_expert, limit_rl):
        policy_agent = self.pi_approx(limit_rl=limit_rl)
        empirical_feature = self.empirical_feature_count(episodes_expert)
        expected_feature = self.expected_feature_count(policy_agent, episodes_expert)
        # print("empirical: ", empirical_feature)
        # print("expected: ", expected_feature)
        grad = empirical_feature - expected_feature

        self.R_approx.weight = self.R_approx.weight + self.alpha * grad
        self.report_R.append([self.R_approx(s) for s in MDP.STATE_SET])

        
    def fit(self, limit_irl=10, limit_rl=10, n=1, verbose=False):
        for i in range(limit_irl):
            if verbose:
                print("\rIRL % d" % i, end="")
            episodes = []
            for _ in range(n):
                # TODO: check!
                episode = self.env.sample_trajectory(self.policy_expert, end=False)
                # episode = self.env.sample_trajectory(self.policy_expert, end=True)
                episodes.append(episode)
            self.update(episodes, limit_rl=limit_rl)
        if verbose:
            print()