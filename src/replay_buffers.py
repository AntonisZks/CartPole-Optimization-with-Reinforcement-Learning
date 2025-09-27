import numpy as np
from collections import deque
import torch as T


class ReplayBufferMemory:
    def __init__(self, capacity=10000, device='cpu'):
        self.device = device
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
        return (
            T.Tensor(np.array(states)).to(self.device),
            T.LongTensor(np.array(actions)).to(self.device),
            T.Tensor(np.array(rewards)).to(self.device),
            T.Tensor(np.array(next_states)).to(self.device),
            T.Tensor(np.array(dones)).to(self.device)
        )
    
    def __len__(self):
        return len(self.memory)
