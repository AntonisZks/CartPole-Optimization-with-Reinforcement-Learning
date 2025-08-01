class Trainer():
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def train_agent(self, num_episodes):
        scores = []
        for episode in range(num_episodes):
            observation, info = self.env.reset(seed=42)
            done = False
            total_score = 0

            while not done:
                action = self.agent.choose_action(observation)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                total_score += reward
                observation = next_observation

            scores.append(total_score)
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}: Total Reward: {total_score}")

        return scores
