import gymnasium as gym
from agents import Agent
from utils import set_seed
import torch as T


def run_env_showcase(env: gym.Env, epochs: int) -> tuple[list[float], int, int]:
    """
    Runs the CartPole-v1 environment with random actions for a specified number of epochs.
    Collects and prints statistics about the performance, including average score, wins, and fails.
    Finally, it plots the scores obtained from running the environment with random actions.
    
    Args:
        env (gym.Env): The CartPole-v1 environment instance.
        epochs (int): Number of epochs to run the environment.
        
    Returns:
        tuple: A tuple containing:
            - list of scores obtained in each epoch,
            - number of wins,
            - number of fails.
    """
    
    epochs_scores = []
    wins, fails = 0, 0
    
    # Run the environment for the specified number of epochs
    for epoch in range(1, epochs+1):
        env.reset()
        done = False
        score, total_steps = 0, 0
        
        while not done:
            action = env.action_space.sample() # Chose a random action from 0 and 1
            observation, reward, terminated, truncated, _ = env.step(action)
            score += reward
            total_steps += 1
            
            # Check if the episode is done, either by termination or truncation
            if terminated or truncated:
                observation, _ = env.reset()
                done = True

                # Count wins and fails based on the episode outcome
                if terminated and total_steps < epochs:
                    fails += 1
                else:
                    wins +=1
        
        epochs_scores.append(score)
    
    return epochs_scores, wins, fails


def test_dqn_trained_model(agent: Agent, env_path: str, test_epochs: int = 200) -> tuple[list[float], int, int]:
    """
    Tests a trained DQN agent on the CartPole-v1 environment for a specified number of epochs.
    Loads the trained model, runs the environment, and collects scores, wins, and fails.

    Args:
        agent (Agent): The DQN agent to be tested.
        env_path (str): Path to the trained model file.
        test_epochs (int, optional): Number of epochs to test the agent. Defaults to 200.

    Returns:
        tuple[list[float], int, int]: A tuple containing:
            - list of scores obtained in each test epoch,
            - number of wins,
            - number of fails.
    """    
    
    # Load the trained model for testing
    model = T.load(env_path)
    agent.policy_network.load_state_dict(model)
    agent.policy_network.eval()

    test_epochs_scores = []
    wins, fails = 0, 0

    env = gym.make('CartPole-v1')
    set_seed(42, env=env)  # Ensure reproducibility during testing

    for epoch in range(1, test_epochs+1):
        observation, info = env.reset(seed=42 if epoch == 1 else None)
        done = False
        score, total_steps = 0, 0
        
        while not done:
            action = agent.choose_action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            score += reward
            total_steps += 1
            
            if terminated or truncated:
                observation, _ = env.reset(seed=42 if epoch == 1 else None)
                done = True

                if terminated and total_steps < test_epochs:
                    fails += 1
                else:
                    wins +=1
        
        test_epochs_scores.append(score)

    env.close()

    dqn_scores = test_epochs_scores
    dqn_wins, dqn_fails = wins, fails

    return dqn_scores, dqn_wins, dqn_fails
