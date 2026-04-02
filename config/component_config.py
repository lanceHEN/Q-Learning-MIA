"""
This module provides several dataclasses for modular configuration of different
components in the MIA pipeline.
"""

from dataclasses import dataclass, field

import gymnasium as gym

@dataclass
class QLearnerConfig:
    """
    Stores config info for QLearner.
    """
    env: gym.Env
    alpha: float = 0.0001
    buffer_size: int = 10000
    buffer_batch_size: int = 32
    verbose: int = 0
    discount_factor: float = 0.999
    epsilon: float = 1
    
@dataclass
class DQNConfig:
    """
    Stores config info for DQN implementations.
    """
    env: gym.Env
    policy: str = "MlpPolicy"
    verbose: int = 0
    alpha: float = 0.0001
    learning_starts: int = 10000
    batch_size: int = 64
    buffer_size: int = 50000
    exploration_fraction: float = 0.3
    exploration_final_eps: float = 0.05
    optimize_memory_usage: bool = True

@dataclass
class DataOracleConfig:
    """
    Stores config info for DataOracle.
    """
    env: gym.Env
    verbose: int = 0

@dataclass
class RandomDataOracleConfig(DataOracleConfig):
    """
    Stores config info for RandomDataOracle.
    """
    pass

@dataclass
class QLearnerDataOracleConfig(DataOracleConfig):
    """
    Stores config info for QLearnerDataOracle.
    """
    q_learner_config: QLearnerConfig = field(default_factory=QLearnerConfig)
    learning_starts: int = 1000
    decay_rate: float = 0.999
    random_seed: int = 1

@dataclass
class DQNDataOracleConfig(DataOracleConfig):
    dqn_config: DQNConfig = field(default_factory=DQNConfig)

@dataclass
class TrainerOracleConfig:
    env: gym.Env
    alpha: float = 0.0001
    discount_factor: float = 0.999
    verbose: int = 0
    
@dataclass
class QLearnerTrainerOracleConfig(TrainerOracleConfig):
    """
    Stores config info for QLearnerTrainerOracle.
    """
    q_learner_config: QLearnerConfig = field(default_factory=QLearnerConfig)

@dataclass
class DQNTrainerOracleConfig(TrainerOracleConfig):
    dqn_config: DQNConfig = field(default_factory=DQNConfig)

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