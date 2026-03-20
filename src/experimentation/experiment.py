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
    QLearnerDataOracleConfig,
    ExperimentRunnerConfig
)

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))
from model import (
    QLearnerDataOracle,
    TrainerOracle,
    MIAClassifier
)

class ExperimentRunner:
    """
    The ExperimentRunner class is used to run basic tests on training an MIA
    model to see how well it can perform.
    """
    
    
    def __init__(self, config: ExperimentRunnerConfig):
        """
        Initializes an ExperimentRunner with the given config.
        """
        self.env = config.env
        self.T_max = config.T_max
        self.seed = config.seed
        self.data_oracle_config = config.data_oracle_config
        self.data_oracle_train_timesteps = config.data_oracle_train_timesteps
        self.n_trajectories = config.n_trajectories
        self.train_external_split = config.train_external_split
        self.trainer_oracle_config = config.trainer_oracle_config
        self.trainer_oracle_train_timesteps = config.trainer_oracle_train_timesteps
        self.fp_rate = config.fp_rate
        self.mia_train_test_split = config.mia_train_test_split
    
    def run_experiment(self):
        """
        Runs a basic MIA experiment, constructing data and train oracles and
        performing an MIA attack.
        """
        # Construct data oracle
        data_oracle = QLearnerDataOracle(self.data_oracle_config)

        # Train for some timesteps
        data_oracle.train(self.data_oracle_train_timesteps)

        # Generate trajectories

        trajectories = data_oracle.generate_trajectories(self.n_trajectories, self.T_max, self.seed)
        
        n_train = round(self.train_external_split * self.n_trajectories)
        n_external = self.n_trajectories - n_train

        train_trajectories = trajectories[:n_train]
        external_trajectories = trajectories[n_train:]

        trainer_oracle = TrainerOracle(self.trainer_oracle_config)

        # Run Q learning on train trajectories
        trainer_oracle.train(train_trajectories, self.trainer_oracle_train_timesteps)

        # Initialize and use MIA Classifier
        mia_classifier = MIAClassifier(trainer_oracle)
        
        # Train/test split for train/external
        train_trajectories_fit = train_trajectories[:round(self.mia_train_test_split * n_train)]
        train_trajectories_nonfit = train_trajectories[round(self.mia_train_test_split * n_train):]
        
        external_trajectories_fit = external_trajectories[:round(self.mia_train_test_split * n_external)]
        external_trajectories_nonfit = external_trajectories[round(self.mia_train_test_split * n_external):]

        mia_classifier.fit(train_trajectories_fit, external_trajectories_fit, fp_rate=self.fp_rate)

        print(f"Learned eta: {mia_classifier.eta}")

        # Get predictions on trajs not used to fit
        train_predictions = mia_classifier.predict_memberships(train_trajectories_nonfit)
        external_predictions = mia_classifier.predict_memberships(external_trajectories_nonfit)

        print(f"Train trajectories accuracy: {np.mean(train_predictions)}")
        print(f"External trajectories accuracy: {1 - np.mean(external_predictions)}")
        
def main():
    env = gym.make("Taxi-v3")
    T_max = 100
    verbose = 1
    seed = 1
    
    data_oracle_config = QLearnerDataOracleConfig(
        env=env,
        verbose=verbose,
        random_seed=seed
    )
    
    trainer_oracle_config = TrainerOracleConfig(
        verbose=verbose
    )
    
    experiment_config = ExperimentRunnerConfig(
        env=env,
        data_oracle_config=data_oracle_config,
        trainer_oracle_config=trainer_oracle_config,
        T_max=T_max,
        seed=seed,
        data_oracle_train_timesteps=10000,
        n_trajectories=10000,
        train_external_split=0.5,
        trainer_oracle_train_timesteps=100000,
        fp_rate=0.05
    )
    
    experiment_runner = ExperimentRunner(experiment_config)
    
    experiment_runner.run_experiment()
    
if __name__ == "__main__":
    main()