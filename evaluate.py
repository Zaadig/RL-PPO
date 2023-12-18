import gym
import torch
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

    env.close()  # Close the environment after evaluation
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"Average Reward: {avg_reward}")

if __name__ == "__main__":
    model_path = 'ppo_model_episode_199.pth'
    evaluate_agent(model_path)
