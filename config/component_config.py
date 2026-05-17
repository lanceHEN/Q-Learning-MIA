"""
This module provides several dataclasses for modular configuration of different
components in the MIA pipeline.
"""

from dataclasses import dataclass, field
import gymnasium as gym
from stable_baselines3 import DQN
from typing import Callable, Union, Tuple

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
class CustomFixedPolicyDataOracleConfig(Da):
    """
    Stores config info for CustomFixedPolicyDataOracle.
    """
    action_selector: Callable[[Union[int, Tuple]], Union[int, Tuple]]
    

@dataclass
class QLearnerDataOracleConfig(DataOracleConfig):
    """
    Stores config info for QLearnerDataOracle.
    """
    q_learner: QLearner = field(default_factory=QLearner)
    learning_starts: int = 1000
    decay_rate: float = 0.999
    random_seed: int = 1

@dataclass
class DQNDataOracleConfig(DataOracleConfig):
    dqn: DQN = field(default_factory=DQN)

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
    data_oracle: DataOracle = field(default_factory=DataOracle)
    q_learner: QLearner = field(default_factory=QLearner)

@dataclass
class DeepTrainerOracleConfig(TrainerOracleConfig):
    dqn: DQN = field(default_factory=DQN)
    
@dataclass
class DeepOfflineTrainerOracleConfig(DeepTrainerOracleConfig):
    data_oracle: DataOracle = field(default_factory=DataOracle)
    
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