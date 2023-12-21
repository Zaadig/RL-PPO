import matplotlib.pyplot as plt
import json

file_path = 'training_log_old.json'

with open(file_path, 'r') as file:
    data = json.load(file)

rewards = data['episode_reward']

episodes = list(range(1, len(rewards) + 1))

plt.figure(figsize=(10, 6))
plt.plot(episodes, rewards, marker='o')
plt.title('Evolution of Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)
plt.savefig('training_reward_log.png')
plt.show()
