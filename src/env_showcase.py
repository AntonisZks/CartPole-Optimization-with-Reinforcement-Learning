import gymnasium as gym
from testing import run_env_showcase
from plotting import plot_default_scores


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    observations = env.observation_space.shape[0]
    actions = env.action_space.n

    print("\nCartPole Environment Details:")
    print(f"Observations: {observations}, Actions: {actions}\n")


    epochs = 200
    print(f"Running the default environment for {epochs} epochs...\n")
    scores, wins, fails = run_env_showcase(env, epochs)

    env.close()

    print(f"Average Score: {sum(scores)/len(scores)}")
    print(f"Wins: {wins}, Fails: {fails}")

    # Plot the scores obtained from running the environment with random actions
    plot_default_scores(scores)
