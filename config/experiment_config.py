"""
This module provides several dataclasses for configuration of experimentation
details.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Callable
from dataclasses import dataclass, field

import gymnasium as gym

from .component_config import SARSAMIAConfig

if TYPE_CHECKING:
    from src.model import TrainerOracle, MIAClassifier

@dataclass
class ExperimentRunnerConfig:
    """
    Stores config info for ExperimentRunner.
    """
    experiment_name: str
    env: gym.Env
    trainer_oracle: TrainerOracle
    mia_classifier: MIAClassifier
    T_max: int = 100
    seed: int = 1
    n_trajectories: int = 10000
    train_external_split: float = 0.5
    trainer_oracle_epochs: int = 500
    fp_rate: float = 0.05
    mia_train_test_split: float = 0.8
    n_eval_trajectories: int = 2000
    raw_results_dir: Optional[str] = "data/raw_results" 
    save_transition_data: bool = True
    save_plots: bool = True  # per-run diagnostic plots (score/residual histograms, ROC curve, policy heatmap); turn off for sweeps 