import numpy as np


class Bandit(object):
    def __init__(self, epsilon, n_arms):
        self.test_1 = 0.
        self.epsilon = epsilon
        self.arms = np.zeros(n_arms)
        self.n_arms = n_arms
        self.choses = np.zeros(n_arms)
        self.q = np.zeros(n_arms)

    def reset(self):
        self.test_1 = 0.
        self.arms = np.zeros(self.arms.shape)
        self.choses = np.zeros(self.n_arms)
        self.q = np.zeros(self.n_arms)

    def choose_action(self):
        if np.random.uniform() > self.epsilon:
            self.test_1 += 1
            return np.argmax(self.q)
        else:
            return np.random.choice(range(len(self.q)), 1)[0]

    def update(self, arm, reward):
        self.arms[arm] += reward
        self.choses[arm] += 1.
        self.q[arm] = self.arms[arm] / self.choses[arm]




