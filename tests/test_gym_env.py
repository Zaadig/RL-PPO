import gym

def test_atari_env(env_name='ALE/Breakout-v5', num_episodes=5, render_mode='human'):
    env = gym.make(env_name, render_mode=render_mode)
    env.reset()

    for episode in range(num_episodes):
        env.reset()
        done = False
        while not done:
            env.render()  # Render the environment
            action = env.action_space.sample()  # Taking a random action
            step_result = env.step(action)
            done = step_result[2]  # Extract the 'done' flag

    env.close()

if __name__ == "__main__":
    test_atari_env()
