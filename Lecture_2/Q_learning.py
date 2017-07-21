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
env = gym.wrappers.Monitor(env, 'q_learnig', force=True)
Q = np.random.rand(env.observation_space.n, env.action_space.n)
R = np.zeros((env.observation_space.n, env.action_space.n))
N = np.zeros((env.observation_space.n, env.action_space.n))
actions = range(env.action_space.n)
gamma = 0.99

n_episodes = 10000

# TO DO:
for j in range(n_episodes):
    done = False

    policy = epsilon_greedy_policy(Q, epsilon=0.1*(0.9**j), actions=actions)
    # episode = []
    state = env.reset()
    ### Generate sample episode
    while not done:
        action = policy(state)
        new_state, reward, done, _ = env.step(action)
        N[state, action] += 1
        if not done:
            q = np.max(Q[new_state])

        else:
            q = 0
        Q[state, action] += (reward + gamma * q - Q[state, action]) * 0.85
        state = new_state


env.close()

gym.upload('q_learnig', api_key=API_KEY)
