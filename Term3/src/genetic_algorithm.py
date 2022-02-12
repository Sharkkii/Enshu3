import numpy as np
import random
from function import *
from environment import *
from programming import *

# NOTE: Genetic Algorithm
# FIXME: see ReguralizedGA (spec. has been changed)
class GeneticAlgorithm:

    MUTATION_RATE = 0.05

    def __init__(self, Gene, Environment, P=10, R=5, T=100):
        self.population = []
        self.fitness = []
        self.Gene = Gene
        self.Environment = Environment
        self.P = P
        self.R = R
        self.T = T
        self.report = []
        
    def show(self):
        for p in self.population:
            p.show()
            
    def initialize(self, init=None):
        self.population = [self.Gene.generate() for _ in range(self.P)]
        if init is not None:
            for idx in range(self.P):
                self.population[idx].set(init[idx])
        self.fitness = self.evaluate(self.population)
            
    def evaluate(self, population):
        fitness = self.Environment.evaluate(population, array=True)
        return fitness
    
    def fit(self, init=None, threshold=1e-8, h=5, verbose=True, verbose_interval=100):

        self.initialize(init)

        history = np.ones(h)
        for epoch in range(self.T):
            
            print("\r%d" % epoch, end="")
            
            old_generation = self.population
            new_generation = []

            # reproduce
            new_generation += old_generation

            # crossover
            for _ in range(self.R):
                p, q = np.random.choice(old_generation, 2, replace=False)
                new_generation += list(self.Gene.crossover(p, q))

            # mutate
            flag = np.random.randint(100, size=len(new_generation)) < GeneticAlgorithm.MUTATION_RATE * 100
            for idx in range(len(new_generation)):
                if flag[idx]:
                    new_generation[idx] = self.Gene.mutate(new_generation[idx])
            
            # evaluate
            fitness = self.evaluate(new_generation)
            p_fitness = fitness / np.sum(fitness)
            # p_fitness = softmax(fitness)

            # select
            self.population = list(np.random.choice(new_generation, size=self.P, p=p_fitness, replace=False))

            # convergence judgement
            old_fitness = self.fitness
            new_fitness = self.evaluate(self.population)

            # measure by AVERAGE
            old = np.mean(old_fitness)
            new = np.mean(new_fitness)

            # measure by MAXIMUM
            # old = np.max(old_fitness)
            # new = np.max(new_fitness)
            
            history = np.roll(history, -1)
            history[-1] = abs((new - old) / new)
            if (np.all(history < threshold)):
                break
            else:
                self.fitness = new_fitness

            # FIXME:
            # if verbose:
            #     if (epoch % verbose_interval == 0):
            #         fitness_max = np.max(self.fitness)
            #         fitness_argmax = np.argmax(self.fitness)
            #         print("\rmaximum: ", self.population[fitness_argmax].bits, fitness_max)
            
            self.report.append([[p.get(), f] for (p,f) in zip(self.population, self.fitness)])
        
        if verbose:
            print()
        print("total epoch: %d" % (epoch+1))


# NOTE: Reguralized Genetic Algorithm
# NOTE: check equivalency by implementing "random" key
class RegularizedGeneticAlgorithm:

    def __init__(self, Generator, Environment, P=10, Q=5, T=100, top=5):
        self.population = []
        self.fitness = []
        self.fingerprint = []
        self.best = []
        self.best_fingerprint = []
        self.Generator = Generator
        self.Environment = Environment
        self.P = P
        self.Q = Q
        self.T = T
        self.top = top
        self.report = []
        
    def show(self):
        for p in self.population:
            p.show()
            
    def initialize(self, init=None):
        self.population = [self.Generator() for _ in range(self.P)]
        if init is not None:
            for idx in range(self.P):
                self.population[idx].set(init[idx])
        self.fitness = self.evaluate(self.population)
        self.fingerprint = list(np.random.randint(low=0, high=1e7, size=self.P))
        self.best, indices = top_n(zip(self.population, self.fitness), n=self.top, return_indices=True)
        self.best_fingerprint = [self.fingerprint[index] for index in indices]

            
    def evaluate(self, population):
        fitness = []
        for p in population:
            fitness.append(self.Environment.evaluate(p))
        return fitness
    
    def fit(self, init=None, threshold=1e-8, h=5, early_stop=False, verbose=True, verbose_interval=100):

        self.initialize(init)

        history = np.ones(h)
        for epoch in range(self.T):
            
            print("\r%d" % epoch, end="")

            # select (remove the oldest individual)
            _ = self.population.pop(0)
            _ = self.fingerprint.pop(0)

            # select (tournament selection)
            individuals = list(np.random.choice(self.population, self.Q, replace=False))
            fitness = self.evaluate(individuals)
            idx = np.argmax(fitness)
            new_individual = individuals[idx]

            # mutate
            new_individual = self.Generator.mutate(new_individual)
            self.population.append(new_individual)
            self.fingerprint.append(np.random.randint(low=0, high=1e7, size=1))

            # convergence judgement
            old_fitness = self.fitness
            new_fitness = self.evaluate(self.population)

            # measure by AVERAGE
            # old = np.mean(old_fitness)
            # new = np.mean(new_fitness)

            # measure by MAXIMUM
            # old = np.max(old_fitness)
            # new = np.max(new_fitness)
            
            # history = np.roll(history, -1)
            # history[-1] = abs((new - old) / new)
            # if (np.all(history < threshold) and early_stop):
            #     break
            # else:
            #     self.fitness = new_fitness

            self.fitness = new_fitness

            # DEBUG:
            # for idx in range(self.P):
            #     if (abs(self.fitness[idx]) > 1e8):
            #         print("score: ", self.fitness[idx])
            #         self.population[idx].show()

            # restore best-n
            for idx in range(self.P):
                if self.fingerprint[idx] not in self.best_fingerprint: 
                    self.best.append((self.population[idx], self.fitness[idx]))
                    self.best_fingerprint.append(self.fingerprint[idx])
            self.best, indices = top_n(self.best, n=self.top, return_indices=True)
            self.best_fingerprint = [self.best_fingerprint[index] for index in indices]
            
            self.report.append([[p.get(), f] for (p,f) in zip(self.population, self.fitness)])
        
        print("total epoch: %d" % (epoch+1))
