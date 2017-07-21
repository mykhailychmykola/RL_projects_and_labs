import numpy as np

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor


class Estimator():
    def __init__(self, env):
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor()
            state = PolynomialFeatures().fit_transform(StandardScaler().fit_transform(env.reset())).reshape((1, -1))
            model.partial_fit([state[0]], [0])
            self.models.append(model)

    def predict(self, s, a=None):
        scaler = StandardScaler()
        features = PolynomialFeatures().fit_transform(scaler.fit_transform(s)).reshape((1, -1))[0]

        if not a:
            # print(np.array([m.predict([features]) for m in self.models]))
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]

    def update(self, s, a, y):
        features = PolynomialFeatures().fit_transform(StandardScaler().fit_transform(s)).reshape((1, -1))[0]
        self.models[a].partial_fit([features], [y])
