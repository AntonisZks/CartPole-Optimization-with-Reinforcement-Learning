from replay_buffers import ReplayBufferMemory
import torch as T
import torch.nn as nn
import numpy as np


class Agent():
    def __init__(
            self, gamma, epsilon, epsilon_min, epsilon_decay, lr, input_dims, n_actions, taget_update_freq, 
            memory_size, batch_size, network, device
        ):
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.device = device

        self.action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size
        self.target_update_freq = taget_update_freq
        self.memory_size = memory_size

        # Initialize the Deep Q-Networks and send to device
        self.policy_network = network(input_dims, n_actions).to(self.device)
        self.target_network = network(input_dims, n_actions).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.learning_rate = lr
        self.optimizer = T.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.memory_buffer = ReplayBufferMemory(capacity=self.memory_size, device=self.device)

        self.steps_done = 0  # Track total steps globally

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            state = T.tensor(np.array([observation]), dtype=T.float32).to(self.device)
            with T.no_grad():
                actions = self.policy_network(state)
            best_action = T.argmax(actions).item()
            return best_action

    def optimize(self):
        if len(self.memory_buffer) <= self.batch_size:
            return

        # Sample batch from replay buffer. Batch tensors are already on the appropriate device via ReplayBufferMemory.sample
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory_buffer.sample(self.batch_size)

        # Compute current Q values
        q_values = self.policy_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute target Q values using target network
        with T.no_grad():
            max_next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)

        # Calculate loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
