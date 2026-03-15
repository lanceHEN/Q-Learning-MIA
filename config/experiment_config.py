"""
This module provides several dataclasses for configuration of experimentation
details.
"""

from dataclasses import dataclass

import gymnasium as gym

@dataclass
class ExperimentRunnerConfig:
    """
    Stores config info for ExperimentRunner.
    """
    env: gym.Env
    data_oracle_config: object
    trainer_oracle_config: object
    T_max: int = 100
    seed: int = 1
    data_oracle_train_timesteps: int = 10000
    n_trajectories: int = 10000
    train_external_split: float = 0.5
    trainer_oracle_train_timesteps: int = 100000
    fp_rate: float = 0.05