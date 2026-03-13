import sys
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import numpy as np

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
from config import (
    TrainerOracleConfig
)

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))
from model import (
    RandomDataOracle,
    TrainerOracle
)

# Make env
env = gym.make("Taxi-v3")

T_max = 100

seed = 1

# Construct data oracle
data_oracle = RandomDataOracle(env)

# Generate trajectories
n_trajectories = 10

trajectories = data_oracle.generate_trajectories(n_trajectories, T_max, seed)

train_trajectories = trajectories[:n_trajectories // 2]
external_trajectories = trajectories[n_trajectories // 2:]

# Initialize Trainer Oracle
trainer_oracle_config = TrainerOracleConfig()

trainer_oracle = TrainerOracle(trainer_oracle_config)

# Run Q learning on train trajectories
trainer_oracle.train(train_trajectories, 1000)

# Examine Bellman residuals
train_membership_scores = []
for traj in train_trajectories:
    s = 0
    for (state, action, reward, next_state) in traj:
        max_next_val = trainer_oracle.optimal_state_val(next_state)
        gamma = trainer_oracle.discount_factor
        q_val = trainer_oracle.q_table[state][action]
        
        s += (reward + gamma*max_next_val - q_val)**2

    train_membership_scores.append(s / len(traj))
    
print(f"Mean train trajectory membership score: {np.mean(train_membership_scores)}")

external_membership_scores = []
for traj in external_trajectories:
    s = 0
    for (state, action, reward, next_state) in traj:
        max_next_val = trainer_oracle.optimal_state_val(next_state)
        gamma = trainer_oracle.discount_factor
        q_val = trainer_oracle.q_table[state][action]
        
        s += (reward + gamma*max_next_val - q_val)**2

    external_membership_scores.append(s / len(traj))
    
print(f"Mean external trajectory membership score: {np.mean(external_membership_scores)}")