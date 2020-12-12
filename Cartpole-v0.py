import gym
import numpy as np
import math

class Agent:
    def __init__(self, env):
        self.env = env
        self.n_buckets = (1, 1, 6, 3)
        self.n_actions = env.action_space.n
        self.bound = list(zip(env.observation_space.low, env.observation_space.high))
        self.bound[1] = (-0.5, 0.5)
        self.bound[3] = (-math.radians(50), math.radians(50))
        self.Q_table = np.zeros(self.n_buckets + (self.n_actions,))
        self.epsilon = 1.0
        self.alpha = 0.5
        self.gamma = 0.99

    def discretize(self, observation):
        state = [0] * len(observation)
        for i, s in enumerate(observation):
            low, high = self.bound[i]
            if s < low:
                state[i] = 0
            elif s > high:
                state[i] = self.n_buckets[i] - 1
            else:
                state[i] = int((s - low) / (high - low) * self.n_buckets[i])

        return tuple(state)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])

    def update_table(self, state, action, reward, next_state, ep):
        self.epsilon = self._get_epsilon(ep)
        self.alpha = self._get_epsilon(ep)
        self.Q_table[state + (action,)] += self.alpha * (reward + \
                                                         self.gamma * np.amax(self.Q_table[next_state]) - \
                                                         self.Q_table[state + (action,)])

    def _get_epsilon(self, ep):
        return max(0.01, min(1, 1.0 - math.log10((ep+1)/25)))

    def _get_alpha(self, ep):
        return max(0.01, min(0.5, 1.0 - math.log10((ep+1)/25)))

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    agent = Agent(env)

    for ep in range(250):
        observation = env.reset()
        state = agent.discretize(observation)
        total_reward = 0.0

        for t in range(250):
            # env.render()
            action = agent.choose_action(state)
            observation, reward, done, _ = env.step(action)

            total_reward += reward

            next_state = agent.discretize(observation)
            agent.update_table(state, action, reward, next_state, ep)

            if done:
                print("episode {} finish at {} timestamps total reward: {}".format(ep, t+1, total_reward))
                break
            else:
                state = next_state
    env.close()
