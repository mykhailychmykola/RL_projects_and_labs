import numpy as np
import pickle


class Arm(object):
    def __init__(self, name):
        self.name = name
        self._dict_of_arms = pickle.load(open("stock_data.p", "rb"))

    def step(self, step):
        # print(self._dict_of_arms[self.name][step])
        return self._dict_of_arms[self.name][step]
