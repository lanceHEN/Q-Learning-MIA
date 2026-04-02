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
from stable_baselines3 import DQN



root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
from config import (
    QLearnerTrainerOracleConfig,
    DeepTrainerOracleConfig,
    QLearnerDataOracleConfig,
    ExperimentRunnerConfig,
    SARSAMIAConfig,
    QLearnerConfig,
    DQNDataOracleConfig,
    DeepOnlineTrainerOracleConfig
)

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))
from model import (
    QLearnerDataOracle,
    QLearnerTrainerOracle,
    DeepTrainerOracle,
    MIAClassifier,
    SARSAMIA,
    QLearner,
    DQNDataOracle,
    DeepOnlineTrainerOracle
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
        self.trainer_oracle = config.trainer_oracle
        self.mia_classifier = config.mia_classifier
        self.T_max = config.T_max
        self.seed = config.seed
        self.n_trajectories = config.n_trajectories
        self.train_external_split = config.train_external_split
        self.trainer_oracle_train_timesteps = config.trainer_oracle_train_timesteps
        self.fp_rate = config.fp_rate
        self.mia_train_test_split = config.mia_train_test_split
        
    def first_experiment_half(self):
        """
        Runs experiment up to producing train/external trajectories, which
        are themselves train-test split for an MIA classifier.
        """
        n_train = round(self.n_trajectories * self.train_external_split)
        n_external = self.n_trajectories - n_train
        
        # Learn
        train_trajectories, external_trajectories = self.trainer_oracle.train(self.trainer_oracle_train_timesteps, n_train, n_external, self.T_max, self.seed)
        
        # Train/test split for train/external
        train_trajectories_fit = train_trajectories[:round(self.mia_train_test_split * n_train)]
        train_trajectories_nonfit = train_trajectories[round(self.mia_train_test_split * n_train):]
        
        external_trajectories_fit = external_trajectories[:round(self.mia_train_test_split * n_external)]
        external_trajectories_nonfit = external_trajectories[round(self.mia_train_test_split * n_external):]
        
        return train_trajectories_fit, train_trajectories_nonfit, external_trajectories_fit, external_trajectories_nonfit
    
    def second_experiment_half(self,
                                train_trajectories_fit,
                                train_trajectories_nonfit,
                                external_trajectories_fit,
                                external_trajectories_nonfit,
                                fp_rate):
        """
        Runs rest of experiment, training the MIA classifier for some fp rate.
        """
        #print(train_trajectories_fit)
        #print(external_trajectories_fit)
        
        self.mia_classifier.fit(train_trajectories_fit, external_trajectories_fit, fp_rate=fp_rate)

        # print(f"Learned eta: {mia_classifier.eta}")

        # Get predictions on trajs not used to fit
        train_predictions = self.mia_classifier.predict_memberships(train_trajectories_nonfit)
        external_predictions = self.mia_classifier.predict_memberships(external_trajectories_nonfit)
        
        preds = np.concatenate((train_predictions, external_predictions))
        
        labels = np.concatenate((np.ones((len(train_predictions))), np.zeros((len(external_predictions)))))

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        
        return accuracy, precision, recall, self.mia_classifier.eta
        
    def run_experiment(self) -> Tuple[float, float, float, float]:
        """
        Runs a basic MIA experiment, constructing data and train oracles and
        performing an MIA attack according to the given config. Returns accuracy,
        precision, and recall scores, along with the learned LRT threshold.
        """
        train_trajectories_fit, train_trajectories_nonfit, external_trajectories_fit, external_trajectories_nonfit = self.first_experiment_half()
        
        return self.second_experiment_half(train_trajectories_fit,
                                            train_trajectories_nonfit,
                                            external_trajectories_fit,
                                            external_trajectories_nonfit,
                                            self.fp_rate)      
    

def test_fp_rates(experiment_runner: ExperimentRunner, fp_rate_list: List[float] = [0.05,0.1,0.15,0.2,0.25]) -> pd.DataFrame:
    
    train_trajectories_fit, train_trajectories_nonfit, external_trajectories_fit, external_trajectories_nonfit = experiment_runner.first_experiment_half()
        
    accuracies = []
    precisions = []
    recalls = []
    etas = []
        
    for fp_rate in fp_rate_list:
        accuracy, precision, recall, eta = experiment_runner.second_experiment_half(train_trajectories_fit,
                                            train_trajectories_nonfit,
                                            external_trajectories_fit,
                                            external_trajectories_nonfit,
                                            fp_rate)    
            
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        etas.append(eta)
        
    return pd.DataFrame({
        "FP Rate":fp_rate_list,
        "Accuracy":accuracies,
        "Precision":precisions,
        "Recall":recalls,
        "Eta":etas
    })   
    
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
                    experiment_config.train_external_split_list = train_external_split
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
    
    # Construct data oracle
    '''
    data_oracle = QLearnerDataOracle(
        QLearnerDataOracleConfig(
            env=env,
            q_learner=QLearner(QLearnerConfig(
                env=env
            ))
        )
    )
    
    # Train for some timesteps
    data_oracle.train(10000)
    
    
    trainer_oracle = QLearnerTrainerOracle(
        QLearnerTrainerOracleConfig(
            env=env,
            data_oracle=data_oracle,
            q_learner=QLearner(QLearnerConfig(
                env=env
            ))
        )
    )
    '''
    
    # Use deep trainer oracle
    trainer_oracle = DeepOnlineTrainerOracle(
        DeepOnlineTrainerOracleConfig(
            env=env,
            dqn=DQN(policy="MlpPolicy",
                       env=env,
                       verbose=verbose,
                       learning_rate=0.0001,
                       learning_starts=1000,
                       exploration_fraction=0.3,
                       exploration_final_eps=0.05,
                       batch_size=64,
                       buffer_size=50000,
                       optimize_memory_usage=False)
        )
    )
    
    mia_classifier = MIAClassifier(
        trainer_oracle
    )
    
    experiment_config = ExperimentRunnerConfig(
        env=env,
        trainer_oracle=trainer_oracle,
        mia_classifier=mia_classifier,
        T_max=T_max,
        seed=seed,
        n_trajectories=500,
        train_external_split=0.5,
        trainer_oracle_train_timesteps=2000000,
        fp_rate=0.05,
        mia_train_test_split=0.8
    )
    
    experiment_runner = ExperimentRunner(experiment_config)

    table = test_fp_rates(experiment_runner, fp_rate_list = [.05])
    table.to_csv("data/tables/hyperparam_results_taxi_dqn_test.csv")
    
    print(table)
    
if __name__ == "__main__":
    main()