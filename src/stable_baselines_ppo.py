from stable_baselines3 import PPO
import gym

env = gym.make("CartPole-v1")
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/")
TIMESTEPS = 10000

for i in range(1, 10):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f'../models/{TIMESTEPS * i}.pth')
