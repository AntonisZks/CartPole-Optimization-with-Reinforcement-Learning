class Trainer():
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def train_agent(self, num_episodes):
        score_per_episode = []
        loss_per_episode = []
        epsilon_per_episode = []

        for episode in range(num_episodes):
            observation, info = self.env.reset()
            episode_score = 0
            episode_loss = 0
            done = False

            while not done:
                action = self.agent.choose_action(observation)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.memory_buffer.add(observation, action, reward, next_observation, float(done))

                observation = next_observation
                episode_score += reward

                loss = self.optimize()
                if loss is not None:
                    episode_loss += loss

                self.steps_done += 1
                if self.steps_done % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.policy_network.state_dict())

            epsilon_per_episode.append(self.epsilon)
            score_per_episode.append(episode_score)
            loss_per_episode.append(episode_loss)

            self.decay_epsilon()

            # Logging progress every ~10% of total episodes
            if episode == 0 or episode == num_episodes - 1 or episode % max(1, num_episodes // 10) == 0:
                if episode == 0 or episode == num_episodes - 1: print(f'Episode {episode+1}:', end=' ')
                else: print(f'Episode {episode}: ', end=' ')
                print(f"Score: {episode_score}, Epsilon: {self.epsilon:.2f}")

        return {
            "scores": score_per_episode, 
            "losses": loss_per_episode, 
            "epsilons": epsilon_per_episode
        }
