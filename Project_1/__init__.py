import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from Project_1.Bandit import Bandit
from Project_1.Arm import Arm


def test_bandit(bandit, arm_func, n_episodes, n_steps, stocks):
    chosen_arms = np.zeros((n_episodes, n_steps))
    chs = np.zeros((n_episodes, n_steps))
    test_1 = 0
    arms = arm_func(stocks)
    for episode in range(n_episodes):
        ep_reward = 0
        bandit.reset()
        for step in range(n_steps):
            chosen_arm = bandit.choose_action()
            chosen_arms[episode, step] = chosen_arm
            reward = arms[chosen_arm].step(step)
            chs[episode, step] = reward
            ep_reward += reward
            bandit.update(chosen_arm, reward)
        test_1 += bandit.test_1
        if episode % 100 == 0:
            print("Reward in episode {}: {}".format(episode, ep_reward))
    print('Test 1: {}'.format(test_1 / (n_episodes * n_steps)))
    return n_steps, np.mean(np.cumsum(chs, axis=1), axis=0)


if __name__ == "__main__":
    stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "FB", "JNJ", "XOM", "JPM", "WFC", "BAC", "GE", "T", "PG", "WMT", "V"]
    numbers_of_arms = len(stocks)
    def generate_arms(stocks):
        return [Arm(name=name) for name in stocks]
    n_steps1, reward1 = test_bandit(bandit=Bandit(0.1, numbers_of_arms), arm_func=generate_arms,
                                    n_episodes=1000, n_steps=1090, stocks=stocks)
    n_steps2, reward2 = test_bandit(bandit=Bandit(0, numbers_of_arms), arm_func=generate_arms,
                                    n_episodes=1000, n_steps=1090, stocks=stocks)
    n_steps3, reward3 = test_bandit(bandit=Bandit(0.01, numbers_of_arms), arm_func=generate_arms,
                                    n_episodes=1000, n_steps=1090, stocks=stocks)
    plt.figure(figsize=(15, 15))
    plt.plot(np.arange(n_steps1), reward1)
    plt.plot(np.arange(n_steps3), reward3)
    plt.plot(np.arange(n_steps2), reward2)
    plt.legend(('epsilon = 0.1', 'epsilon = 0.01', 'epsilon = 0'))
    plt.savefig('normal.png')
    plt.show()

