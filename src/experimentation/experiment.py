import sys
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
from config import (
    TrainerOracleConfig,
    QLearnerDataOracleConfig
)

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))
from model import (
    QLearnerDataOracle,
    TrainerOracle,
    MIAClassifier
)


# Make env
env = gym.make("Taxi-v3")

T_max = 100

seed = 1

# Construct data oracle
data_oracle_config = QLearnerDataOracleConfig(
    env=env
)

data_oracle = QLearnerDataOracle(data_oracle_config)

# Train for some timesteps
data_oracle.train(10000)

# Generate trajectories
n_trajectories = 10000

trajectories = data_oracle.generate_trajectories(n_trajectories, T_max, seed)

train_trajectories = trajectories[:n_trajectories // 2]
external_trajectories = trajectories[n_trajectories // 2:]

# Initialize Trainer Oracle
trainer_oracle_config = TrainerOracleConfig(
    buffer_size=100000
)

trainer_oracle = TrainerOracle(trainer_oracle_config)

# Run Q learning on train trajectories
trainer_oracle.train(train_trajectories, 100000)

# Initialize and use MIA Classifier
mia_classifier = MIAClassifier(trainer_oracle)

mia_classifier.train(train_trajectories, external_trajectories, fp_rate=0.05)

print(f"Learned eta: {mia_classifier.eta}")

# Get predictions
train_predictions = mia_classifier.predict_memberships(train_trajectories)
external_predictions = mia_classifier.predict_memberships(external_trajectories)

print(f"Train trajectories accuracy: {np.mean(train_predictions)}")
print(f"External trajectories accuracy: {1 - np.mean(external_predictions)}")