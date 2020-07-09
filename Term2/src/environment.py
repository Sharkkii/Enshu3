import numpy as np


# Markov Decision Process

class MDP:
    STATE_N = 9; ACTION_N = 4
    STATE_SET = np.arange(9); ACTION_SET = np.arange(4)
    INITIAL_STATE = 0; GOAL_STATE = 8
    VALID_ACTION_SET = np.array([
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 0],
    ])
    
    INITIAL_STATE_TABLE = np.ones(STATE_N) / (STATE_N - 1)
    INITIAL_STATE_TABLE[GOAL_STATE] = 0.0
    INITIAL_DISTRIBUTION = lambda state: MDP.INITIAL_STATE_TABLE[state]
    
    TRANSITION_TABLE = np.zeros((STATE_N, ACTION_N, STATE_N))
    TRANSITION_TABLE[0, 1, 3] = 1
    TRANSITION_TABLE[0, 3, 1] = 1
    TRANSITION_TABLE[1, 2, 0] = 1
    TRANSITION_TABLE[1, 3, 2] = 1
    TRANSITION_TABLE[2, 1, 5] = 1
    TRANSITION_TABLE[2, 2, 1] = 1
    TRANSITION_TABLE[3, 0, 0] = 1
    TRANSITION_TABLE[3, 1, 6] = 1
    TRANSITION_TABLE[3, 3, 4] = 1
    TRANSITION_TABLE[4, 1, 7] = 1
    TRANSITION_TABLE[4, 2, 3] = 1
    TRANSITION_TABLE[5, 0, 2] = 1
    TRANSITION_TABLE[6, 0, 3] = 1
    TRANSITION_TABLE[7, 0, 4] = 1
    TRANSITION_TABLE[7, 3, 8] = 1
    TRANSITION_TABLE[8, 2, 8] = 1

    TRANSITION_FUNCTION = lambda state, action: MDP.TRANSITION_TABLE[state, action]
    
    DISCOUNT_RATE = 0.80
    
    REWARD_TABLE = np.zeros((STATE_N, ACTION_N))
    REWARD_TABLE[7, 3] = 1.0
    REWARD_FUNCTION = lambda state, action: MDP.REWARD_TABLE[state, action]
    
    POLICY_RANDOM = lambda state: MDP.VALID_ACTION_SET[state] / np.sum(MDP.VALID_ACTION_SET[state])
    POLICY_OPTIM_TABLE = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    POLICY_EXPERT_TABLE = np.array([
        [0.00, 0.95, 0.00, 0.05],
        [0.00, 0.00, 0.95, 0.05],
        [0.00, 0.05, 0.95, 0.00],
        [0.05, 0.05, 0.00, 0.90],
        [0.00, 0.95, 0.05, 0.00],
        [1.00, 0.00, 0.00, 0.00],
        [1.00, 0.00, 0.00, 0.00],
        [0.05, 0.00, 0.00, 0.95],
        [0.00, 0.00, 1.00, 0.00],
    ])
    POLICY_OPTIM = lambda state: MDP.POLICY_OPTIM_TABLE[state]
    POLICY_EXPERT = lambda state: MDP.POLICY_EXPERT_TABLE[state]

    T = 1000
    
    def __init__(self, gamma=DISCOUNT_RATE, R=REWARD_FUNCTION):
        self.gamma = gamma
        self.R = R
    
    def choose_action(self, policy, state):
        action = np.random.choice(MDP.ACTION_SET, p=policy(state))
        return action
    
    def get_next_state(self, state, action):
        state = np.random.choice(MDP.STATE_SET, p=MDP.TRANSITION_FUNCTION(state, action))
        return state

    def update_R(self, R):
        self.R = R
    
    # def sample_trajectory(self, policy, init_state=None, init_action=None, sarsa=False, end=False):
    #     # NOTE: initial distribution
    #     trajectory = []
    #     self.set_state(init_state)
    #     self.set_action(init_action)
        
    #     if self.state == MDP.GOAL_STATE: return traejectory
    #     if self.action is None: self.action = self.choose_action(policy)
            
    #     reward = self.R(self.state, self.action)
    #     next_state = self.get_next_state()
    #     trajectory.append((self.state, self.action, reward, next_state))

    #     while next_state != MDP.GOAL_STATE:
    #         self.state = next_state
    #         self.action = self.choose_action(policy)
    #         reward = self.R(self.state, self.action)
    #         next_state = self.get_next_state()
    #         trajectory.append((self.state, self.action, reward, next_state))

    #     if end:
    #         self.state = next_state
    #         self.action = self.choose_action(policy)
    #         reward = self.R(self.state, self.action)
    #         next_state = self.get_next_state()
    #         trajectory.append((self.state, self.action, reward, next_state))

    #     return trajectory

    def sample_trajectory(self, policy, init_state=None, init_action=None, sarsa=False, end=False):
        # NOTE: initial distribution
        trajectory = []
        t = 0
        state = MDP.INITIAL_STATE if init_state is None else init_state
        action = self.choose_action(policy, state) if init_action is None else init_action

        while state != MDP.GOAL_STATE:
            reward = self.R(state, action)
            next_state = self.get_next_state(state, action)
            next_action = self.choose_action(policy, next_state)
            if sarsa:
                trajectory.append((state, action, reward, next_state, next_action))
            else:
                trajectory.append((state, action, reward, next_state))
            state = next_state
            action = next_action

            # truncate with fixed length
            t += 1
            if t > MDP.T:
                break

        if end:
            reward = self.R(state, action)
            next_state = self.get_next_state(state, action)
            next_action = self.choose_action(policy, next_state)
            if sarsa:
                trajectory.append((state, action, reward, next_state, next_action))
            else:
                trajectory.append((state, action, reward, next_state))

        return trajectory