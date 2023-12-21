import gym
import torch
import matplotlib.pyplot as plt
from agents.ppo_agent import Agent
from config.config import Config
from utils.helpers import preprocess_observation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_agent(model_path, num_episodes=100):
    # Environment set up
    env = gym.make(Config.ENV_NAME)
    env.seed(Config.SEED)

    # Initializing the agent
    agent = Agent(env, Config)
    agent.model.load_state_dict(torch.load(model_path, map_location=device))

    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        state = preprocess_observation(state)
        episode_reward = 0
        done = False

        while not done:
            env.render()  # Render the game screen
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            state = preprocess_observation(state)
            episode_reward += reward

            if done:
                total_rewards.append(episode_reward)
                print(f"Episode: {episode + 1}, Reward: {episode_reward}")

                 # Plotting and saving the rewards
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, episode + 2), total_rewards, marker='o')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title('Rewards per Episode')
                plt.savefig('episode_rewards.png')
                plt.close()

    env.close()  # Close the environment after evaluation
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"Average Reward: {avg_reward}")

if __name__ == "__main__":
    model_path = 'models/ppo_model_episode_600.pth'
    evaluate_agent(model_path)
