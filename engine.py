from IPython.display import clear_output
from time import sleep
import numpy as np
import random
import gym
import matplotlib.pyplot as plt


class QLearner:
    def __init__(self):
        self.env = gym.make("Taxi-v3").env
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))

        self.alpha = 0.2
        self.gamma = 0.5
        self.epsilon = 0.1

    def train(self, max_epochs=30000, print_interval=100,
              alpha=None, gamma=None, epsilon=None):

        # change variable if needed
        self.alpha = alpha or self.alpha
        self.gamma = gamma or self.gamma
        self.epsilon = epsilon or self.epsilon

        scatter_x = []
        scatter_y = []
        scatter_e = []

        for epoch in range(max_epochs):
            current_state = self.env.reset()

            while True:

                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[current_state])

                current_q = self.q_table[current_state, action]
                next_state, reward, done, _ = self.env.step(action)
                max_q = np.max(self.q_table[next_state])
                self.q_table[current_state, action] = current_q + self.alpha * (reward + self.gamma * max_q - current_q)

                current_state = next_state
                if done:
                    break

            if epoch > 0 and epoch % print_interval == 0:
                y, e = self.evaluate()
                scatter_x.append(epoch)
                scatter_y.append(y)
                scatter_e.append(e)

                print(f"Epoch number: {epoch}")

        print("Training completed.")
        plt.errorbar(scatter_x, scatter_y, scatter_e, linestyle='None', marker='^')

        plt.show()

    def evaluate(self):

        max_steps = 100
        epochs = 10
        rewards = np.zeros(epochs)

        for epoch in range(epochs):
            state = self.env.reset()

            total_reward = 0

            step = 0
            while True:
                action = np.argmax(self.q_table[state])
                state, reward, done, _ = self.env.step(action)

                total_reward += reward#(self.gamma**step) * reward

                if done or step >= max_steps:
                    rewards[epoch] = total_reward
                    break

                step += 1

        return rewards.mean(), rewards.std()

    def simulate(self, state=None):
        images = []
        state = self.env.reset()

        max_steps = 100
        step = 0
        while True:
            action = np.argmax(self.q_table[state])
            state, reward, done, _ = self.env.step(action)
            images.append(self.env.render(mode="ansi"))
            if done or step >= max_steps:
                break
            step += 1

        for image in images:
            print(image)
            sleep(1)
if __name__ == "__main__":
    agent = QLearner()
    # %%time
    agent.train()
    agent.simulate()
    pass