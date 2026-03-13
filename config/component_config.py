"""
This module provides several dataclasses for modular configuration of different
components in the MIA pipeline.
"""

from dataclasses import dataclass

import gymnasium as gym

@dataclass
class QLearnerConfig:
    """
    Stores config info for QLearner.
    """

    env: gym.Env
    discount_factor: float = 0.99
    verbose: int = 0
    buffer_size: int = 10000
    buffer_batch_size: int = 32
    learning_starts: int = 1000
    epsilon: float = 0.99
    decay_rate: float = 0.99
    random_seed: int = 1