# agents/model.py


import torch
import torch.optim as optim
from .model import PPOAgent
from collections import deque
import numpy as np
import torch.nn.functional as F

class Agent:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        # Setup the PPO agent
        self.model = PPOAgent(input_channels=1,
                              hidden_size=config.HIDDEN_SIZE,
                              num_actions=env.action_space.n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        # Rollout buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        # print("State shape in choose_action:", state.shape)
        with torch.no_grad():
            action_probs, _ = self.model(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float().unsqueeze(0)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        done = torch.tensor(done)

        # Evaluate action
        with torch.no_grad():
            action_probs, value = self.model(state)
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(action)

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def learn(self):
        R = 0
        returns = deque()

        # Compute returns
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.config.GAMMA * R
            returns.appendleft(R)

        # Normalize returns
        returns = torch.tensor(list(returns)).float()
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Convert lists to tensors
        states = torch.stack([s.squeeze(0) for s in self.states])  # Remove batch dimension before stacking
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)

        # PPO update
        for _ in range(self.config.PPO_EPOCHS):
            action_probs, state_values = self.model(states)
            dist = torch.distributions.Categorical(action_probs)

            # New log probabilities and entropy
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Compute ratio
            ratios = torch.exp(new_log_probs - old_log_probs.detach())

            # Compute surrogate loss
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config.PPO_EPSILON, 1 + self.config.PPO_EPSILON) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(state_values, returns)
            loss = policy_loss + self.config.CRITIC_DISCOUNT * value_loss - self.config.ENTROPY_BETA * entropy.mean()

            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
            self.optimizer.step()

        # Clear the buffer
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()

    def train(self):
        for episode in range(self.config.NUM_EPISODES):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.store_transition(state, action, reward, next_state, done)
                state = next_state

                if done:
                    self.learn()
