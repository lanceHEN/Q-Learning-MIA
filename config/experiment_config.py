"""
This module provides several dataclasses for configuration of experimentation
details.
"""

from dataclasses import dataclass

import gymnasium as gym

from .component_config import SARSAMIAConfig

@dataclass
class ExperimentRunnerConfig:
    """
    Stores config info for ExperimentRunner.
    """
    env: gym.Env
    data_oracle_config: object
    trainer_oracle_config: object
    deep_data_oracle: bool = False
    deep_trainer_oracle: bool = False
    sarsa_attacker: bool = False
    sarsa_config: SARSAMIAConfig = None
    T_max: int = 100
    seed: int = 1
    data_oracle_train_timesteps: int = 10000
    n_trajectories: int = 10000
    train_external_split: float = 0.5
    trainer_oracle_train_timesteps: int = 100000
    fp_rate: float = 0.05
    mia_train_test_split: float = 0.8