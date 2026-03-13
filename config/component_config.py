"""
This module provides several dataclasses for modular configuration of different
components in the MIA pipeline.
"""

from dataclasses import dataclass

import gymnasium as gym

@dataclass
class TrainerOracleConfig:
    """
    Stores config info for TrainreOracle.
    """

    discount_factor: float = 0.99
    verbose: int = 0
    buffer_size: int = 10000
    buffer_batch_size: int = 32