import torch as T
import torch.nn as nn
import numpy as np
import random
from collections import deque

from networks import DeepQNetwork_CartPole


class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, n_actions, device):
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.device = device

        self.action_space = [i for i in range(n_actions)]
        self.batch_size = 64
        self.target_update_freq = 10
        self.memory_size = 10000

        # Initialize the Deep Q-Networks
        self.policy_network = DeepQNetwork_CartPole(input_dims, n_actions)
        self.target_network = DeepQNetwork_CartPole(input_dims, n_actions)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.learning_rate = lr
        self.optimizer = T.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=self.memory_size)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            best_action = np.random.choice(self.action_space)
        else:
            state = T.Tensor([observation]).to(self.device)
            actions = self.policy_network(state)
            best_action = T.argmax(actions).item()

        return best_action

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = T.Tensor(state_batch).to(self.device)
        action_batch = T.LongTensor(action_batch).to(self.device)
        reward_batch = T.Tensor(reward_batch).to(self.device)
        next_state_batch = T.Tensor(next_state_batch).to(self.device)
        done_batch = T.Tensor(done_batch).to(self.device)

        q_values = self.policy_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()

        with T.no_grad():
            max_next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, num_episodes):
        score_per_episode = []
        steps_done = 0
        for episode in range(num_episodes):
            observation, info = env.reset(seed=42)
            episode_score = 0
            done = False

            while not done:
                action = self.choose_action(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated
                self.memory.append((observation, action, reward, next_observation, done))

                observation = next_observation
                episode_score += reward

                self.optimize()

                if steps_done % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.policy_network.state_dict())

                steps_done += 1

            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
            score_per_episode.append(episode_score)
            
            print(f"Episode {episode + 1}: Score: {episode_score}, Epsilon: {self.epsilon:.2f}")

        return score_per_episode

