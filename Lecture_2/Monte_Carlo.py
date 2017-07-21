import numpy as np
import gym

API_KEY = "sk_yy5S2r1SmOL5ZZcNuL5IA"


def epsilon_greedy_policy(Q, epsilon, actions):
    """ Q is a numpy array, epsilon between 0,1
    and a list of actions"""

    def policy_fn(state):
        if np.random.rand() > epsilon:
            action = np.argmax(Q[state, :])
        else:
            action = np.random.choice(actions)
        return action

    return policy_fn


env = gym.make("FrozenLake-v0")
env = gym.wrappers.Monitor(env, 'first_visit', force=True)
Q = np.random.rand(env.observation_space.n, env.action_space.n)
R = np.zeros((env.observation_space.n, env.action_space.n))
N = np.zeros((env.observation_space.n, env.action_space.n))
actions = range(env.action_space.n)
gamma = 1.0

n_episodes = 5000

# TO DO:
for j in range(n_episodes):
    done = False
    state = env.reset()
    policy = epsilon_greedy_policy(Q, epsilon=1./(j + 1), actions=actions)
    episode = []
    ### Generate sample episode
    while not done:
        action = policy(state)
        new_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        print((state, action, reward))
        state = new_state

    ### NOT RELEVANT FOR SARSA/Q-LEARNING
    sa_in_episode = set([(x[0], x[1]) for x in episode])

    # Find first visit of each s,a in the episode
    for s, a in sa_in_episode:
        first_visit = next(i for i, x in enumerate(episode) if
                           x[0] == s and x[1] == a)

        G = sum(x[2] * (gamma ** i) for i, x in enumerate(episode[first_visit:]))
        R[s, a] += G
        N[s, a] += 1
        Q[s, a] += R[s, a] / N[s, a]

env.close()

gym.upload('first_visit', api_key=API_KEY)
