import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(DQN, self).__init__()
    self.fc1 = nn.Linear(input_dim, 128)
    self.fc2 = nn.Linear(128, 128)
    self.fc3 = nn.Linear(128, output_dim)

  def forward(self, observation):
    x = F.relu(self.fc1(observation))
    x = F.relu(self.fc2(x))
    actions = self.fc3(x)

    return actions

def test_agent(env, policy_net, num_episodes=10, render=True):
  for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
      if render:
          env.render()

      # Convert state to tensor and get action from policy network
      state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
      with torch.no_grad():
          action = policy_net(state_tensor).argmax(dim=1).item()  # Choose action with highest Q-value

      # Take the action in the environment
      next_state, reward, done, _, _ = env.step(action)
      total_reward += reward
      state = next_state

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

  env.close()
    
env = gym.make('CartPole-v1', render_mode='human')
observations = env.observation_space.shape[0]
actions = env.action_space.n

# Call the test function
policy_net = DQN(observations, actions)
policy_net.load_state_dict(torch.load('models/makis.pth'))
policy_net.eval()

test_agent(env, policy_net, num_episodes=10)
