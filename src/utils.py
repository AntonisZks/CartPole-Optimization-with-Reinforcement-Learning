import gymnasium as gym
import random
import torch
import numpy as np
import os


# Hyperparameters
GAMMA = 0.99          # Discount factor
LR = 1e-3             # Learning rate
BATCH_SIZE = 64       # Minibatch size
MEMORY_SIZE = 10000   # Replay buffer size
EPSILON_START = 1.0   # Starting exploration probability
EPSILON_END = 0.01    # Minimum exploration probability
EPSILON_DECAY = 0.995 # Epsilon decay rate
TARGET_UPDATE = 10    # How often to update the target network
    

def set_seed(seed: int = 42, env: gym.Env = None) -> None:
    '''
    Sets the seed for reproducibility across various libraries and the environment.
    
    Args:
        seed (int): The seed value to set. Default is 42.
        env (gym.Env): The Gym environment instance to set the seed for. Default is None.
    
    '''
    
    # Set seeds for random, numpy, and torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set seed for the environment if provided
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Ensure deterministic behavior in torch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Additional settings for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            print("Warning: Could not enable deterministic algorithms in torch.")

        os.environ['PYTHONHASHSEED'] = str(seed)
