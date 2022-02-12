import numpy as np
from ..environment import *
from .utility import *

# actual return
# if no trajectory is given, this may be r.v.

def c_return(env, trajectory=None, t=0):
    if trajectory is None:
        trajectory = env.sample_trajectory(MDP.POLICY_RANDOM)
        
    gamma = 1.0
    c = 0.0
    for s, (_, _, reward, _) in enumerate(trajectory):
        if (s < t):
            continue
        c += gamma * reward
        gamma *= env.gamma
    
    return c


# DONE: above
# TODO: clean this room

# def candidate_transformation(state):
#     mapping = np.array([
#         [0.5, 2.5],
#         [1.5, 2.5],
#         [2.5, 2.5],
#         [0.5, 1.5],
#         [1.5, 1.5],
#         [2.5, 1.5],
#         [0.5, 0.5],
#         [1.5, 0.5],
#         [2.5, 0.5]
#     ])
#     return mapping[state]

# # Monte Carlo Approximation

# def actual_return(trajectory, t, gamma=discount_rate):
#     c = 0.0
#     gamma_n = 1.0
#     T = len(trajectory)
#     for k in range(t, T):
#         _, _, reward, _ = trajectory[k]
#         c += gamma_n * reward
#         gamma_n *= gamma
#     return c
    

# class Value_montecarlo:
#     def __init__(self, trajectories):
#         numerator = np.zeros(STATE_N)
#         denominator = np.zeros(STATE_N)
        
#         for trajectory in trajectories:
#             conditioned_actual_return = np.zeros(STATE_N)
#             occupancy_measure = np.zeros(STATE_N)
            
#             for t, (state, _, reward, _) in enumerate(trajectory):
#                 conditioned_actual_return[state] += actual_return(trajectory, t)
#                 occupancy_measure[state] += 1.0
                
#             numerator += conditioned_actual_return
#             denominator += occupancy_measure
            
#         self.data = numerator / denominator
#         self.data = np.where(self.data > 0, self.data, 0)
#         self.method = lambda state: self.data[state]
        
#     def __call__(self, state):
#         return self.method(state)
    
# # Approximation by Bellman operator Approximation

# class Bellman:
#     def __init__(self, trajectories):
#         self.trajectories = trajectories
        
#     def __call__(self, v):
#         numerator = np.zeros(STATE_N)
#         denominator = np.zeros(STATE_N)
        
#         for trajectory in self.trajectories:
#             conditioned_actual_return = np.zeros(STATE_N)
#             occupancy_measure = np.zeros(STATE_N)
            
#             for (state, _, reward, next_state) in trajectory:
#                 conditioned_actual_return[state] += reward + discount_rate * v(next_state)
#                 occupancy_measure[state] += 1.0
                
#             numerator += conditioned_actual_return
#             denominator += occupancy_measure
            
#         self.data = numerator / denominator
#         self.data = np.where(self.data > 0, self.data, 0)
#         self.method = lambda state: self.data[state]
        
#         return self.method

# class Value_bellman:
#     def __init__(self, trajectories, limit=10):
#         v = lambda state: 0.0
#         B = Bellman(trajectories)
#         for _ in range(limit):
#             v = B(v)
            
#         self.method = v
            
#     def __call__(self, state):
#         return self.method(state)
    
# # TD method

# def td_error(sars, V, gamma=discount_rate):
#     state, _, reward, next_state = sars
#     delta = reward + gamma * V(next_state) - V(state)
#     return delta

# class Value_td:
#     def __init__(self, alpha=None):
#         self.data = np.zeros(STATE_N)
#         self.method = lambda state: self.data[state]
#         self.alpha = (lambda t: 1/(1+t)) if alpha is None else alpha
        
#     def __call__(self, state):
#         return self.method(state)
    
#     def update(self, sars, t=0, gamma=discount_rate, alpha=1e-2):
#         delta = td_error(sars, self, gamma)
# #         td_error = reward + gamma * self(next_state) - self(state)
#         state, _, _, _ = sars
#         self.data[state] = self.data[state] + self.alpha(t) * delta
#         self.method = lambda state: self.data[state]
        
# # TD(lambda) method

# def n_step_truncated_return(trajectory, V, t=0, n=np.infty, gamma=discount_rate):
#     T = len(trajectory)
#     c = 0.0
#     gamma_n = 1.0
    
#     next_state, _, _, _ = trajectory[0]
#     for k in range(t, min(t+n, T)):
#         state, _, reward, next_state = trajectory[k]
#         c += gamma_n * reward
#         gamma_n *= gamma
#     if T >= t+n:
#         c += gamma_n * V(next_state)

#     return c

# def n_step_truncated_td_error(trajectory, V, t=0, n=1):
#     state_t, _, _, _ = trajectory[t]
#     return n_step_truncated_return(trajectory, V, t, n) - V(state_t)

# def return_of_forward_view(trajectory, V, t=0, lam=1e-2, gamma=discount_rate):
#     if lam <= 0.0:
#         return 0.0
#     elif lam >= 1.0:
#         return n_step_truncated_return(trajectory, V, t, np.infty, gamma)
#     else:
#         T = len(trajectory)
#         lam_n = 1.0
#         c = 0.0
#         for n in range(T):
#             c += lam_n * n_step_truncated_return(trajectory, V, t, n)
#             lam_n *= lam
#         c *= 1 - lam
#         return c
        
# def td_error_of_forward_view(trajectory, V, t=0, lam=1e-2, gamma=discount_rate):
#     state_t, _, _, _ = trajectory[t]
#     return return_of_forward_view(trajectory, V, t, lam, gamma) - V(state_t)


# # TODO: later
# # class Value_td_forward:
# #     def __init__(self, alpha=None, lam=1e-2):
# #         self.data = np.zeros(STATE_N)
# #         self.method = lambda state: self.data[state]
# #         self.alpha = lambda t: 1/(1+t) if alpha is None else alpha
# #         self.lam = lam
        
# #     def __call__(self, state):
# #         return self.method(state)
    
# #     def update(self, trajectory, t=0, gamma=discount_rate):
# #         td_error = td_error_of_forward_view(trajectory, self, t, self.lam, gamma)
# #         self.data[state] = self.data[state] + self.alpha(t) * td_error
# #         self.method = lambda state: self.data[state]
        
# # TODO: later
# # class Value_td_backward:
# #     def __init__(self):
# #         pass

# # Q learning method

# def td_error_q(sars, Q, gamma=discount_rate):
#     state, action, reward, next_state = sars
#     max_Q = max([Q(next_state, valid_action) for valid_action in get_valid_action_set(next_state)])
#     delta_q = reward + gamma * max_Q - Q(state, action)
#     return delta_q

# def td_error_sarsa(sarsa, Q, gamma=discount_rate):
#     state, action, reward, next_state, next_action = sarsa
#     delta_q = reward + gamma * Q(next_state, next_action) - Q(state, action)
#     return delta_q

# class Q_function:
#     def __init__(self, alpha=None, gamma=discount_rate):
#         self.data = np.zeros((STATE_N, ACTION_N))
#         self.method = lambda state, action: self.data[state, action]
#         self.alpha = (lambda t: 1/(1+t)) if alpha is None else alpha
        
#     def __call__(self, state, action):
#         return self.method(state, action)
    
#     def update(self, sars, t=0, gamma=discount_rate):
#         delta_q = td_error_q(sars, self, gamma)
#         state, action, _, _ = sars
#         self.data[state, action] = self.data[state, action] + self.alpha(t) * delta_q
#         self.method = lambda state, action: self.data[state, action]
        
# class Q_function:
#     def __init__(self, alpha=None, gamma=discount_rate):
#         self.data = np.zeros((STATE_N, ACTION_N))
#         self.method = lambda state, action: self.data[state, action]
#         self.alpha = lambda t: 1/(1+t) if alpha is None else alpha
        
#     def __call__(self, state, action):
#         return self.method(state, action)
    
#     def update(self, sarsa, t=0, gamma=discount_rate):
#         delta_sarsa = td_error_sarsa(sarsa, self, gamma)
#         state, action, _, _, _ = sarsa
#         self.data[state, action] = self.data[state, action] + self.alpha(t) * delta_q
#         self.method = lambda state, action: self.data[state, action]