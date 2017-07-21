import numpy as np


def f(env, weight):
    total_reward = 0.0
    num_run = 10
    for t in range(num_run):
        observation = env.reset()
        for i in range(200):
            action = 1 if np.dot(weight, observation) > 0 else 0
            env.render()
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
    return total_reward / num_run
