# agents/model.py


import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from .model import PPOAgent
from collections import deque
import numpy as np
import torch.nn.functional as F

def compute_gae(rewards, values, gamma, lam):
    # Assuming rewards and values are numpy arrays
    gae = 0
    returns = np.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        returns[t] = gae + values[t]
    return returns


class Agent:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        # Setup the PPO agent
        self.model = PPOAgent(input_channels=1,
                              hidden_size=config.HIDDEN_SIZE,
                              num_actions=env.action_space.n)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device) 

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # Rollout buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # print("State shape in choose_action:", state.shape)
        with torch.no_grad():
            action_probs, _ = self.model(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        done = torch.tensor(done).to(self.device)

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
        # Move tensors to CPU and then convert to NumPy arrays
        rewards = np.array([r.item() for r in self.rewards])
        dones = np.array([d.item() for d in self.dones])

        # Compute state values for all states
        states_tensor = torch.stack(self.states).squeeze(dim=1)
        _, state_values = self.model(states_tensor.to(self.device))
        state_values = state_values.squeeze().cpu().detach().numpy()

        # Append next state value for GAE computation
        next_state_value = 0 if dones[-1] else state_values[-1]
        state_values = np.append(state_values, next_state_value)

        # Compute GAE
        returns = compute_gae(rewards, state_values, self.config.GAMMA, self.config.GAE_LAMBDA)
        returns = torch.tensor(returns).float().to(self.device)

        # Normalize returns
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

            # Update the learning rate at the end of each episode
            self.scheduler.step()

    # def lr_lambda(self, epoch):
    #     # Define a custom lambda function for the scheduler
    #     # This example linearly decreases the learning rate from 2.5e-4 to 1e-5
    #     initial_lr = 2.5e-4
    #     final_lr = 1e-5
    #     total_epochs = self.config.NUM_EPISODES
    #     return final_lr + (initial_lr - final_lr) * (1 - epoch / total_epochs)
