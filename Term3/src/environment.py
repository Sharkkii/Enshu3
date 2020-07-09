import numpy as np
from function import *
from programming import *


# NOTE: a problem must have evaluate method

# maximization of gaussian mixture
class GaussianMixture:
    def __init__(self, D=1, K=2):
        self.D = D
        self.K = K
        self.pi = np.ones((K,)) / K
        self.mu = np.random.rand(K, D)
        self.sigma = np.ones((K,))
        self.lower = -10.0
        self.upper = 10.0
        
    def __call__(self, x, return_vector=False):
            
        N, D = x.shape
        x = x[:, None, :]
        pi = self.pi[None, :]
        mu = self.mu[None, :, :]
        sigma = self.sigma[None, :]
        
        result = np.exp(- np.sum((x - mu)**2, axis=2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)**(D/2)
        result = pi * result
        
        if not return_vector:
            result = np.sum(result, axis=1)
        return result
    
    def get(self):
        return self.pi, self.mu, self.sigma, self.lower, self.upper
    
    def set(self, pi=None, mu=None, sigma=None, lower=None, upper=None):
        if not(pi is None):
            self.pi = pi
        if not(mu is None):
            self.mu = mu
        if not(sigma is None):
            self.sigma = sigma
        if not(lower is None):
            self.lower = lower
        if not(upper is None):
            self.upper = upper
            
    def evaluate(self, x, array=True):
        if array:
            y = np.array([_x.value for _x in x])
            y = y[:, None]
            y = self(y)
            return y
        else:
            y = np.array([x.value])
            y = y[:, None]
            y = self(y)
            return y

    def callback_for_draw_GA(self, idx, x, y, title):
        if (idx > 0):
            plt.cla()
        a = np.linspace(self.lower)
        b = self(a)
        plt.plot(a, b)

        # plt.legend()
        plt.title(title)

        if (idx > 0):
            plt.scatter(x[idx], y[idx], color="green", marker="o")
        

    def draw_GA(self, x, y, title, filename, pause=0):
        figure = plt.figure()

        # augment the last scene
        if (pause > 0):
            x = augment_last(x, pause)
            y = augment_last(y, pause)

        anime = animation.FuncAnimation(figure, self.callback_for_draw_GA, fargs=(x,y,title), interval=100, frames=len(x))
        anime.save(filename+".gif", writer="pillow")
        
        # NOTE: bitsteam
        # if array:
        #     y = np.array([integer_to_float(_x.bits, lower=self.lower, upper=self.upper) for _x in x])
        #     y = y[:, None]
        #     y = self(y)
        #     return y
        # else:
        #     y = integer_to_float(x.bits, lower=self.lower, upper=self.upper)
        #     y = np.array(y)[:, None]
        #     y = self(y)
        #     return y
            

# toy_problem
# R2 <- R0 + R1

class ToyProblem:
    def __init__(self, specification):
        self.specification = specification
        self.simulator = Simulator(specification)
        self.init_pc = 0
        self.questions = [[0 for _ in range(specification.N_REG)]]
        self.answers = [0]
        self.predictor = lambda regs: regs[0].data
        self.metrics = lambda x, y: 0

    def get_qa(self):
        return self.questions, self.answers
    def get_predictor(self):
        return self.predictor
    def get_metrics(self):
        return self.metrics
    
    def set_qa(self, q=None, a=None):
        if (q is not None):
            self.questions = q
        if (a is not None):
            self.answers = a
    def set_predictor(self, predictor):
        self.predictor = predictor
    def set_metrics(self, metrics):
        self.metrics = metrics
    
    # DEBUG:
    def evaluate(self, x):
        # x.show()
        score = []
        for (q, a) in zip(self.questions, self.answers):
            self.simulator.initialize(init_pc=self.init_pc, init_regs=q)
            self.simulator(x)
            y = self.predictor(self.simulator.regs)
            score.append(self.metrics(y, a))
        for i in range(len(score)):
            if score[i] > 0:
                print(score)
                for r in self.simulator.regs:
                    print(r.get())
                assert(False)
        score = np.mean(score)
        return score
