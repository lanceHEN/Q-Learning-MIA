"""
This module provides several dataclasses for modular configuration of different
components in the MIA pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym

from src.model import QLearner, DataOracle

@dataclass
class QLearnerConfig:
    """
    Stores config info for QLearner.
    """
    env: gym.Env
    alpha: float = 0.0001
    buffer_size: int = 100000
    buffer_batch_size: int = 32
    verbose: int = 0
    discount_factor: float = 0.999
    epsilon: float = 1
    state_encoder: object = None

@dataclass
class DataOracleConfig:
    """
    Stores config info for DataOracle.
    """
    env: gym.Env
    verbose: int = 0
    state_encoder: object = None

@dataclass
class RandomDataOracleConfig(DataOracleConfig):
    """
    Stores config info for RandomDataOracle.
    """
    pass

@dataclass
class CustomFixedPolicyDataOracleConfig(DataOracleConfig):
    """
    Stores config info for CustomFixedPolicyDataOracle.
    """
    action_selector: object = None


@dataclass
class QLearnerDataOracleConfig(DataOracleConfig):
    """
    Stores config info for QLearnerDataOracle.
    """
    q_learner: Optional[QLearner] = None
    learning_starts: int = 1000
    decay_rate: float = 0.999
    random_seed: int = 1

@dataclass
class DQNDataOracleConfig(DataOracleConfig):
    learning_rate: float = 0.0005
    learning_starts: int = 1000
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05
    batch_size: int = 64
    buffer_size: int = 100000
    device: str = "auto"

@dataclass
class TrainerOracleConfig:
    env: gym.Env
    alpha: float = 0.0001
    discount_factor: float = 0.999
    verbose: int = 0
    state_encoder: object = None

@dataclass
class QLearnerTrainerOracleConfig(TrainerOracleConfig):
    """
    Stores config info for QLearnerTrainerOracle.
    """
    data_oracle: Optional[DataOracle] = None
    q_learner: Optional[QLearner] = None

@dataclass
class DeepTrainerOracleConfig(TrainerOracleConfig):
    learning_rate: float = 0.0005
    learning_starts: int = 0
    exploration_fraction: float = 0
    exploration_final_eps: float = 0
    batch_size: int = 32
    buffer_size: int = 10000000
    device: str = "auto"

@dataclass
class DeepOfflineTrainerOracleConfig(DeepTrainerOracleConfig):
    data_oracle: Optional[DataOracle] = None

@dataclass
class DeepOnlineTrainerOracleConfig(DeepTrainerOracleConfig):
    pass

@dataclass
class SARSAMIAConfig:
    """
    Stores config info for SARSAMIA.
    """
    alpha: float = 0.0001
    discount_factor: float = 0.999
    n_trajectories: int = 1000
    n_epochs: int = 10
    T_max: int = 100
    seed: int = None
