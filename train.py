# train.py

import gym
import torch
from agents.ppo_agent import Agent
from config.config import Config
from utils.logger import Logger
from utils.helpers import preprocess_observation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_agent():
    # Set up the environment
    env = gym.make(Config.ENV_NAME)

    # Set random seeds for reproducibility
    torch.manual_seed(Config.SEED)
    env.seed(Config.SEED)

    # Initialize the PPO agent
    agent = Agent(env, Config)

    # Initialize the logger
    logger = Logger('training_log.json')

    for episode in range(Config.NUM_EPISODES):
        state = env.reset()[0]
        state = preprocess_observation(state)
        episode_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_observation(next_state)

            agent.store_transition(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                break

        agent.learn()  # Perform learning at the end of each episode

        # Logging
        logger.log('episode_reward', episode_reward)
        if episode % Config.LOG_EVERY == 0:
            print(f'Episode: {episode}, Reward: {episode_reward}')
            logger.save()

        # Save the model periodically and at the end of training
        if episode % Config.SAVE_MODEL_EVERY == 0 or episode == Config.NUM_EPISODES - 1:
            torch.save(agent.model.state_dict(), f'ppo_model_episode_{episode}.pth')

    env.close()

if __name__ == "__main__":
    train_agent()
