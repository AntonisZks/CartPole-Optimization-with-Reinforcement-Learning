from stable_baselines3 import A2C
import gym

env = gym.make("CartPole-v1")
env.reset()

model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/")
TIMESTEPS = 10000

for i in range(1, 10):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f'../models/{TIMESTEPS * i}.pth')
    