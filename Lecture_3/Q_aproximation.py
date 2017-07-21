import gym
import numpy as np
import itertools
import sys
from collections import Counter
from Lecture_3.Estimator import Estimator
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

API_KEY = "sk_yy5S2r1SmOL5ZZcNuL5IA"

env = gym.make("CartPole-v0")
env = gym.wrappers.Monitor(env, 'cartpole', force=True)


def make_epsilon_greedy_policy(estimator, epsilon, normalisator):
    def policy_fn(observation):
        A = np.ones(normalisator, dtype=float) * epsilon / normalisator

        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)

        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def q_learning(env, estimator, num_episodes, discount_factor=0.9, epsilon=0.1, epsilon_decay=0.99):
    counter = []
    for i in range(num_episodes):
        policy = make_epsilon_greedy_policy(
            estimator, epsilon*(0.9**i), env.action_space.n)
        state = env.reset()
        next_action = None
        for t in itertools.count():
            if next_action is None:
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                counter.append(action)
            else:
                action = next_action
                counter.append(action)
            next_state, reward, done, _ = env.step(action)
            reward = -1 if reward == 0 else reward
            q_values_next = estimator.predict(next_state)
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            td_target = reward + discount_factor * q_values_next[next_action]
            estimator.update(state, action, td_target)
            if done:
                break
            state = next_state
    return counter


if __name__ == '__main__':
    estim = Estimator(env)
    counter = q_learning(env, estim, 1000, epsilon=0.5)
    env.close()
    print(Counter(counter))

    gym.upload('cartpole', api_key=API_KEY)
