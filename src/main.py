import gym
import torch as T
import matplotlib.pyplot as plt
import random

from agents import Agent
from trainers import Trainer


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.n

    seed = 42
    random.seed(seed)
    T.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed(seed)

    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    agent = Agent(
        gamma=0.99, 
        epsilon=1.0, 
        lr=0.001, 
        input_dims=input_dims, 
        n_actions=n_actions, 
        device=device
    )
    
    scores = agent.train(env, num_episodes=200)
    env.close()

    plt.plot(scores, color='dodgerblue')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.ylim(0, 600)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')

    plt.grid('grey', linestyle='--', alpha=0.5)
    plt.show()

    # Use tained agent to view the environment
    env = gym.make('CartPole-v1', render_mode='human')
    observation, info = env.reset(seed=seed)
    done = False

    for _ in range(500):
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation = next_observation

    env.close()
