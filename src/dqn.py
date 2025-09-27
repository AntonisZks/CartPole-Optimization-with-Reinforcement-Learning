from plotting import plot_trained_vs_dummy_results
from networks import DeepQNetwork_CartPole
from plotting import plot_training_results
from testing import test_dqn_trained_model
from trainers import Trainer
from agents import Agent
import gymnasium as gym
from utils import *
import torch as T


if __name__ == '__main__':
    
    # Create the CartPole environment and receive its details (observations, actions)
    env = gym.make('CartPole-v1')
    observations = env.observation_space.shape[0]
    actions = env.action_space.n
    
    # Set the device to GPU if available, else CPU
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    print(f"\nStarting DQN training (Device used: {device})\n")
    
    # Initialize the Agent and Trainer with the appropriate parameters
    agent = Agent(
        GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY, LR, observations, actions, 
        TARGET_UPDATE, MEMORY_SIZE, BATCH_SIZE, DeepQNetwork_CartPole, device
    )
    trainer = Trainer(agent, env)

    # Train the agent for a specified number of episodes and collect results
    results = trainer.train_agent(500)
    env.close()

    # Save the trained model, and plot the training results
    T.save(agent.policy_network.state_dict(), 'models/dqn_model.pth')
    print("\nDQN training completed and model saved as 'models/dqn_model.pth'\n")
    plot_training_results(results)

    # Test the trained model and display the results
    test_scores, test_wins, test_fails = test_dqn_trained_model(agent, 'models/dqn_model.pth')
    print(f"\nTesting Results over 200 epochs")
    print(f"Average Score: {sum(test_scores)/len(test_scores)}")
    print(f"Wins: {test_wins}, Fails: {test_fails}\n")

    # Plot comparison between trained agent and random actions