import numpy as np


class Arm(object):
    def __init__(self, p):
        self.p = p

    def step(self):
        return np.random.normal(self.p)
