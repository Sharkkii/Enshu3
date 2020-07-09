import numpy as np

# maze generator

class Maze:
    def __init__(self, M, N, constraint=0):
        self.M = M
        self.N = N
        self.state = 0
        self.coordinate = (0, 0)
        self.state_set = np.arange(M*N)
        self.action_set = np.ones((M*N, 4))
        self.initial_state = 0
        self.goal_state = M*N - 1
        for state in self.state_set:
            if state // N == 0: self.action_set[state, 0] = 0
            if state // N == M-1: self.action_set[state, 1] = 0
            if state % M == 0: self.action_set[state, 2] = 0
            if state % M == N-1: self.action_set[state, 3] = 0
                
        for _ in range(constraint):
            s = np.random.randint(M*N)
            a = np.random.randint(4)
            self.action_set[s, a] = 0
                
    def coordinate2state(self, coordinate):
        y, x = coordinate
        state = y * self.N + x
        return state
    
    def state2coordinate(self, state):
        y = state // self.N
        x = state % self.N
        coordinate = (y, x)
        return coordinate
            
    def move(self, state, action):
        y, x = self.state2coordinate(state)
        if action == 0: y -= 1
        if action == 1: y += 1
        if action == 2: x -= 1
        if action == 3: x += 1
        state = self.coordinate2state((y, x))
        return state
        
    # BFS
    def search(self):
        open_queue = [np.array([self.initial_state])]
        close_queue = []
        
        while len(open_queue) > 0:
            traj = open_queue.pop(0)
            s = traj[-1]
            valid_action_set = np.arange(4)[self.action_set[s] > 0]
            for a in valid_action_set:
                new_s = self.move(s, a)
                if new_s not in traj:
                    new_traj = np.append(traj, new_s)
                    if new_s == self.goal_state:
                        close_queue.append(new_traj)
                    else:
                        open_queue.append(new_traj)
        
        return close_queue
        

def main():
    maze = Maze(3,3, constraint=0)
    maze.search()


if __name__ == "__main__":
    main()