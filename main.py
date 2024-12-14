import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')
states = env.observation_space.shape[0]
actions = env.action_space.n

episodes = 100
for episode in range(1, episodes+1):
  state, info = env.reset()
  done = False
  score = 0

  while not done:
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    score += reward
    
    if terminated or truncated:
      observation, info = env.reset()
      done = True
    
  print('Episode:{} Score:{}'.format(episode, score))

env.close()
