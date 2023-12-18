# utils/helpers.py

import cv2
import numpy as np

def preprocess_observation(observation, new_size=(84, 84), grayscale=True):
    """
    Preprocess the observation from the Atari environment.

    Args:
    - observation (np.array): The original observation from the environment.
    - new_size (tuple): The new size of the observation after resizing.
    - grayscale (bool): Whether to convert the observation to grayscale.

    Returns:
    - np.array: The preprocessed observation.
    """

    if grayscale:
        # Convert RGB to BGR
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        # Then convert BGR to grayscale
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)

    observation = cv2.resize(observation, new_size, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to be between 0 and 1
    observation = observation / 255.0

    return np.expand_dims(observation, axis=0)  # Add a channel dimension for batch

