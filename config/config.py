
class Config:
    # Environment settings
    ENV_NAME = "Breakout-v0"  # Name of the environment to train on
    NUM_EPISODES = 1000           # Total number of training episodes
    RENDER_ENV = True             # Whether to render the environment
    RENDER_EVERY = 50             # Render the environment every N episodes
    LOG_EVERY = 10                # Logging interval

    # Neural network architecture
    HIDDEN_SIZE = 256             # Number of nodes in each hidden layer
    NUM_HIDDEN_LAYERS = 2         # Number of hidden layers

    # Hyperparameters for PPO
    LEARNING_RATE = 3e-4          # Learning rate
    GAMMA = 0.99                  # Discount factor for reward
    GAE_LAMBDA = 0.95             # GAE lambda parameter
    PPO_EPSILON = 0.2             # Clip parameter for PPO
    CRITIC_DISCOUNT = 0.5         # Discount factor for critic updates
    ENTROPY_BETA = 0.01           # Entropy bonus coefficient
    MAX_GRAD_NORM = 0.5           # Maximum gradient norm

    # Training settings
    UPDATE_EVERY = 2048           # How often to update the network
    MINI_BATCH_SIZE = 64          # Size of the mini-batch
    PPO_EPOCHS = 10               # Number of epochs to update the network
    SAVE_MODEL_EVERY = 50         # Number of episodes for setting a checkpoint

    # Other settings
    SEED = 42                     # Random seed for reproducibility
