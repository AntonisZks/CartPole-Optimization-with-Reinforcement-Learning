import gym

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode='human')
    observation, info = env.reset(seed=42)

    for _ in range(500):
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    print("Environment closed successfully.")
