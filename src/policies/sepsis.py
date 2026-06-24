"""
Stochastic expert policy for the ICU-Sepsis environment, loaded from a
pre-computed policy table.
"""
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_DEFAULT_POLICY_PATH = Path(__file__).parent / "expertPolicy.csv"


def make_expert_policy(policy_path: Path = _DEFAULT_POLICY_PATH) -> Callable[[int], int]:
    """
    Returns a stochastic expert policy for the ICU-Sepsis environment.

    The policy table is a CSV with one row per state and one column per action,
    where each entry is an unnormalized action probability.
    """
    table = pd.read_csv(policy_path, header=None).values

    def policy(state: int) -> int:
        probs = table[state].astype(float)
        probs /= probs.sum()
        return np.random.choice(len(probs), p=probs)

    return policy
