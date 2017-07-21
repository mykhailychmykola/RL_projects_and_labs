import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Lecture_1.Bandit import Bandit
from Lecture_1.Arm import Arm
from Lecture_1.SoftmaxBandit import SoftmaxBandit


def test_bandit(bandit, arm_func, n_episodes, n_steps):
    chosen_arms = np.zeros((n_episodes, n_steps))
    chs = np.zeros((n_episodes, n_steps))
    test_1 = 0
    for episode in range(n_episodes):

        ep_reward = 0
        bandit.reset()
        arms = arm_func(bandit.n_arms)
        for step in range(n_steps):
            chosen_arm = bandit.choose_action()
            chosen_arms[episode, step] = chosen_arm
            reward = arms[chosen_arm].step()
            chs[episode, step] = reward
            ep_reward += reward
            bandit.update(chosen_arm, reward)
        test_1 += bandit.test_1
        if episode % 100 == 0:
            print("Reward in episode {}: {}".format(episode, ep_reward))
    print("Test 1: How often does our algorithm pick the best new_action?")
    print('Test 1: {}'.format(test_1 / (n_episodes * n_steps)))
    sns.heatmap(chosen_arms)
    # plt.show()
    # plt.plot(np.arange(n_steps), np.sum(chosen_arms, axis=0))
    # plt.show()
    # plt.close('all')

    print("Test 2: TODO - Track the average reward over time")
    print("Test 3: TODO - Track cumulative reward")
    print(chosen_arms)
    return n_steps, np.mean(chs, axis=0)


if __name__ == "__main__":
    numbers_of_arms = 10


    def generate_arms(number_of_arms):
        arms = []
        for i in range(numbers_of_arms):
            arms.append(Arm(np.random.normal()))
        return arms


    n_steps1, reward1 = test_bandit(bandit=Bandit(0.1, numbers_of_arms), arm_func=generate_arms,
                                    n_episodes=1000, n_steps=2000)
    n_steps2, reward2 = test_bandit(bandit=Bandit(0, numbers_of_arms), arm_func=generate_arms,
                                    n_episodes=1000, n_steps=2000)
    n_steps3, reward3 = test_bandit(bandit=Bandit(0.01, numbers_of_arms), arm_func=generate_arms,
                                    n_episodes=1000, n_steps=2000)
    plt.figure(figsize=(15, 15))
    plt.plot(np.arange(n_steps1), reward1)
    plt.plot(np.arange(n_steps3), reward3)
    plt.plot(np.arange(n_steps2), reward2)

    plt.legend(('epsilon = 0.1', 'epsilon = 0.01', 'epsilon = 0'))
    plt.savefig('normal.png')
    plt.show()

    n_steps1, reward1 = test_bandit(bandit=SoftmaxBandit(0.1, numbers_of_arms, 0.1), arm_func=generate_arms,
                                    n_episodes=1000, n_steps=2000)
    n_steps2, reward2 = test_bandit(bandit=SoftmaxBandit(0, numbers_of_arms, 0.1), arm_func=generate_arms,
                                    n_episodes=1000, n_steps=2000)
    n_steps3, reward3 = test_bandit(bandit=SoftmaxBandit(0.01, numbers_of_arms, 0.1), arm_func=generate_arms,
                                    n_episodes=1000, n_steps=2000)
    plt.figure(figsize=(15, 15))
    plt.plot(np.arange(n_steps1), reward1)
    plt.plot(np.arange(n_steps3), reward3)
    plt.plot(np.arange(n_steps2), reward2)

    plt.legend(('epsilon = 0.1', 'epsilon = 0.01', 'epsilon = 0'))
    plt.savefig('softmax.png')
    plt.show()
