import numpy as np


class Evolution_algorithms(object):
    def __init__(self, f, env):
        self.f = f
        self.env = env

    def nes(self, n_population=10, sigma=0.1, alpha=0.001, n_iter=30):
        weights = np.random.rand(4) * 2 - 1
        for i in range(n_iter):
            population = np.random.randn(n_population, 4)
            rewards = np.zeros(n_population)
            for j in range(n_population):
                w_temp = weights + sigma * population[j]
                rewards[j] = self.f(self.env, w_temp)
            new = (rewards - np.mean(rewards)) / np.std(rewards)
            weights += alpha / (n_population * sigma) * np.dot(population.T, new)

    def cem(self, batch=25, frac=0.2, n_iter=200):
        mean = np.zeros(4)
        sigma = np.ones(4)
        for i in range(n_iter):
            w = np.random.randn(batch, 4) * sigma + mean
            reward = np.zeros(batch)
            for j in range(batch):
                reward[j] = self.f(self.env, w[j])
            keep = int(batch * frac)
            sit = w[np.argsort(reward)[batch - keep:]]
            mean = np.mean(sit, axis=0)
            sigma = np.std(sit, axis=0)
        return sit
