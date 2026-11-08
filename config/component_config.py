"""
This module provides several dataclasses for modular configuration of different
components in the MIA pipeline.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.model import QLearner, DataOracle, TrainerOracle

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
    unseen_training_val: float = 0 # what value to assign to unseen (s,a) pairs (per submodel) in training
    unseen_inference_val: float = 0.0 # wwhat value to assign to unseen (s,a) pairs (per submodel) in inference

@dataclass
class DataOracleConfig:
    """
    Stores config info for DataOracle.
    """
    env: gym.Env
    verbose: int = 0
    state_encoder: object = None

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
    alpha_decay_rate: float = 1.0  # multiplies the Q-learner's learning rate after every training step; 1 = constant
    random_seed: int = 1
    tau: float = 0.0

@dataclass
class DQNDataOracleConfig(DataOracleConfig):
    learning_rate: float = 0.0005
    learning_starts: int = 1000
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05
    batch_size: int = 64
    buffer_size: int = 100000
    device: str = "auto"
    policy: str = "MlpPolicy"
    policy_kwargs: dict = None
    seed: int = None
    target_update_interval: int = 1000
    gamma: float = 0.99
    tau: float = 0.0

@dataclass
class TrainerOracleConfig:
    env: gym.Env
    discount_factor: float = 0.999
    verbose: int = 0
    state_encoder: object = None

@dataclass
class QLearnerTrainerOracleConfig(TrainerOracleConfig):
    """
    Stores config info for QLearnerTrainerOracle. The learning rate and discount factor
    are q_learner's; discount_factor must match q_learner's.
    """
    data_oracle: Optional[DataOracle] = None
    q_learner: Optional[QLearner] = None

@dataclass
class DeepTrainerOracleConfig(TrainerOracleConfig):
    learning_rate: float = 0.0005
    batch_size: int = 32
    buffer_size: int = 10000000
    device: str = "auto"
    policy: str = "MlpPolicy"
    data_oracle: Optional[DataOracle] = None
    cql_alpha: float = 0.0 # 0 = standard DQN, >0 = CQL penalty
    target_bounds: Optional[Tuple[float, float]] = None # clamp bootstrap targets to the range of returns; None = no clamping
    target_updates: int = 30
    net_arch: Optional[List[int]] = None # hidden layer sizes of the Q-network; None = SB3's default [64, 64]

@dataclass
class DeepDiscretizingTrainerOracleConfig(DeepTrainerOracleConfig):
    """
    DQN whose released Q function is a table over a fixed grid of states. Discrete state
    spaces use their own states as the grid. Noise-free; use as an EnsemblingTrainerOracle
    sub-model for a DP guarantee.
    """
    points: Optional[np.ndarray] = None # (n_cells, obs_dim) public points to release Q values for; None on a continuous space queries the network at the exact observation (no table, so no DP release)

@dataclass
class DPSGDTrainerOracleConfig(TrainerOracleConfig):
    """
    Offline DQN (optionally CQL) trained with DP-SGD via Opacus, with one trajectory as
    the privacy unit. discount_factor is used both for the TD targets and by the attack.
    The number of epochs comes from the experiment runner (trainer_oracle_epochs).
    """
    data_oracle: Optional[DataOracle] = None
    hidden_sizes: Tuple[int, ...] = (64, 64)
    learning_rate: float = 1e-3
    batch_size: int = 64                 
    target_updates: int = 30
    cql_alpha: float = 0.0               
    target_bounds: Optional[Tuple[float, float]] = None # clamp bootstrap targets to the range of returns; None = no clamping
    trajectory_loss: str = "sum" # how to aggregate over individual transition losses
    max_grad_norm: float = 1.0          
    epsilon: Optional[float] = 10.0      
    delta: float = 1e-5                 
    neighboring_relation: str = "replace_one" 
    accountant: str = "prv" # Opacus accountant, used only with add_or_remove_one; replace_one uses dp_accounting's PLD accountant
    device: str = "cpu"

@dataclass
class EnsemblingTrainerOracleConfig(TrainerOracleConfig):
    data_oracle: Optional[DataOracle] = None
    k: int = 2
    trainer_factory: Optional[Callable[[], TrainerOracle]] = None  # builds a fresh, independent sub-oracle
    q_bounds: Tuple[float, float] = (0.0, 1.0)
    epsilon: Optional[float] = 1.0 
    delta: float = 1e-5

@dataclass
class DPAVIVITrainerOracleConfig(TrainerOracleConfig):
    """
    Noisy counts -> pessimistic value iteration (DP-APVI, Qiao & Wang 2022, Algorithm 1),
    with Gaussian noise for (epsilon, delta)-DP. failure_prob is the confidence bounds'
    failure probability (Algorithm 1's delta), which is separate from the DP delta. Set
    discount_factor to match the learner, since the attack reads it from here.
    """
    data_oracle: Optional[DataOracle] = None
    epsilon: Optional[float] = 1000.0
    delta: float = 1e-5 
    T_max: int = 50   
    public_support: Optional[object] = None 
    reward_table: Optional[object] = None  # public reward model, S x S' or S x A x S'; None: the environment's own (ICU-Sepsis's _r_mat)
    C: float = 2.0            
    failure_prob: float = 0.05        
    value_bound: Optional[float] = None 
    penalty_scale: float = 1.0    
    consistency_solver: str = "closed_form"
