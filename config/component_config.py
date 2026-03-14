"""
This module provides several dataclasses for modular configuration of different
components in the MIA pipeline.
"""

from dataclasses import dataclass

import gymnasium as gym

@dataclass
class TrainerOracleConfig:
    """
    Stores config info for TrainerOracle.
    """
    discount_factor: float = 0.99
    verbose: int = 0
    buffer_size: int = 10000
    buffer_batch_size: int = 32
    
@dataclass
class RandomDataOracleConfig:
    """
    Stores config info for RandomDataOracle.
    """
    env: gym.Env

@dataclass
class QLearnerDataOracleConfig:
    """
    Stores config info for QLearnerDataOracle.
    """
    env: gym.Env
    discount_factor: float = 0.99
    verbose: int = 0
    learning_starts: int = 1000
    buffer_size: int = 10000
    buffer_batch_size: int = 32
    epsilon: float = 0.99
    decay_rate: float = 0.99
    random_seed: int = 1
    