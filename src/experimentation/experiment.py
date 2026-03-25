import sys
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import ale_py



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
    
    def run_experiment(self) -> Tuple[float, float, float, float]:
        """
        Runs a basic MIA experiment, constructing data and train oracles and
        performing an MIA attack according to the given config. Returns accuracy,
        precision, and recall scores, along with the learned LRT threshold.
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

        # print(f"Learned eta: {mia_classifier.eta}")

        # Get predictions on trajs not used to fit
        train_predictions = mia_classifier.predict_memberships(train_trajectories_nonfit)
        external_predictions = mia_classifier.predict_memberships(external_trajectories_nonfit)
        
        preds = np.concatenate((train_predictions, external_predictions))
        
        labels = np.concatenate((np.ones((len(train_predictions))), np.zeros((len(external_predictions)))))

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        
        return accuracy, precision, recall, mia_classifier.eta
    
def test_hyperparams(experiment_config: ExperimentRunnerConfig,
                     n_trajectories_list: List[int] = [500],
                     train_external_split_list: List[float] = [0.5],
                     trainer_oracle_train_timesteps_list: List[int] = [1000000],
                     fp_rate_list: List[float] = [0.05]) -> pd.DataFrame:
    """
    Produces a table of train/test classification metrics and learned LRT thresholds
    for each possible configuration from the given lists. Overrides the values given
    in experiment_config for each test.
    """
    # The following lists keep track of combinations and results to make the table.
    n_traj = []
    splits = []
    steps = []
    rates = []
    
    accuracies = []
    precisions = []
    recalls = []
    etas = []
    
    for n_trajectories in n_trajectories_list:
        for train_external_split in train_external_split_list:
            for trainer_oracle_train_timesteps in trainer_oracle_train_timesteps_list:
                for fp_rate in fp_rate_list:
                    experiment_config.n_trajectories = n_trajectories
                    experiment_config.train_external_split_list = train_external_split_list
                    experiment_config.trainer_oracle_train_timesteps = trainer_oracle_train_timesteps
                    experiment_config.fp_rate = fp_rate
                    
                    experiment_runner = ExperimentRunner(experiment_config)
    
                    accuracy, precision, recall, eta = experiment_runner.run_experiment()
                    
                    n_traj.append(n_trajectories)
                    splits.append(train_external_split)
                    steps.append(trainer_oracle_train_timesteps)
                    rates.append(fp_rate)
                    
                    accuracies.append(accuracy)
                    precisions.append(precision)
                    recalls.append(recall)
                    etas.append(eta)
                    
    return pd.DataFrame({
        "N. Trajectories":n_traj,
        "Train/External Traj Split":splits,
        "Trainer Oracle Timesteps":steps,
        "FP Rate":rates,
        "Accuracy":accuracies,
        "Precision":precisions,
        "Recall":recalls,
        "Eta":etas
    })  
        
def main():
    
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-ram-v5")
    T_max = float('inf')
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
        n_trajectories=500,
        train_external_split=0.5,
        trainer_oracle_train_timesteps=1000000,
        fp_rate=0.05
    )
    
    table = test_hyperparams(experiment_config,
                     n_trajectories_list = [250],
                     train_external_split_list = [0.5],
                     trainer_oracle_train_timesteps_list = [10000000],
                     fp_rate_list = [0.05,0.1,0.15,0.2,0.25])
    table.to_csv("hyperparam_results_pong_2.csv")
    
    print(table)
    
if __name__ == "__main__":
    main()