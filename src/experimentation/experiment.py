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
    QLearnerTrainerOracleConfig,
    DQNTrainerOracleConfig,
    QLearnerDataOracleConfig,
    ExperimentRunnerConfig,
    SARSAMIAConfig,
    QLearnerConfig,
    DQNConfig,
    DQNDataOracleConfig
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
    DQNDataOracle
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
        self.data_oracle_config = config.data_oracle_config
        self.trainer_oracle_config = config.trainer_oracle_config
        self.deep_data_oracle = config.deep_data_oracle
        self.deep_trainer_oracle = config.deep_trainer_oracle
        self.sarsa_attacker = config.sarsa_attacker
        self.sarsa_config = config.sarsa_config
        self.T_max = config.T_max
        self.seed = config.seed
        self.data_oracle_train_timesteps = config.data_oracle_train_timesteps
        self.n_trajectories = config.n_trajectories
        self.train_external_split = config.train_external_split
        self.trainer_oracle_train_timesteps = config.trainer_oracle_train_timesteps
        self.fp_rate = config.fp_rate
        self.mia_train_test_split = config.mia_train_test_split
        
    def _first_experiment_half(self):
        """
        Runs experiment up to producing trainer oracle along with train/external trajectories, which
        are themselves train-test split for an MIA classifier.
        """
        
        
        # Construct data oracle
        if self.deep_data_oracle:
            data_oracle = DQNDataOracle(self.data_oracle_config)
        else:
            data_oracle = QLearnerDataOracle(self.data_oracle_config)

        # Train for some timesteps
        data_oracle.train(self.data_oracle_train_timesteps)

        # Generate trajectories

        trajectories = data_oracle.generate_trajectories(self.n_trajectories, self.T_max, self.seed)
        
        n_train = round(self.train_external_split * self.n_trajectories)
        n_external = self.n_trajectories - n_train

        train_trajectories = trajectories[:n_train]
        external_trajectories = trajectories[n_train:]

        if self.deep_trainer_oracle:
            trainer_oracle = DeepTrainerOracle(self.trainer_oracle_config)
        else:
            trainer_oracle = QLearnerTrainerOracle(self.trainer_oracle_config)

        # Run Q learning on train trajectories
        trainer_oracle.train(train_trajectories, self.trainer_oracle_train_timesteps)
        
        # Train/test split for train/external
        train_trajectories_fit = train_trajectories[:round(self.mia_train_test_split * n_train)]
        train_trajectories_nonfit = train_trajectories[round(self.mia_train_test_split * n_train):]
        
        external_trajectories_fit = external_trajectories[:round(self.mia_train_test_split * n_external)]
        external_trajectories_nonfit = external_trajectories[round(self.mia_train_test_split * n_external):]
        
        return trainer_oracle, train_trajectories_fit, train_trajectories_nonfit, external_trajectories_fit, external_trajectories_nonfit
    
    def _second_experiment_half(self,
                                trainer_oracle,
                                train_trajectories_fit,
                                train_trajectories_nonfit,
                                external_trajectories_fit,
                                external_trajectories_nonfit,
                                fp_rate):
        """
        Runs rest of experiment, training the MIA classifier for some fp rate.
        """
        
        # Initialize and use MIA Classifier
        if self.sarsa_attacker:
            mia_classifier = SARSAMIA(trainer_oracle, self.sarsa_config.alpha, self.sarsa_config.discount_factor)
            # Have to fill Q table
            mia_classifier.train(self.sarsa_config.n_trajectories, self.sarsa_config.n_epochs, self.sarsa_config.T_max, self.sarsa_config.seed)
        else:
            mia_classifier = MIAClassifier(trainer_oracle)
        
        mia_classifier.fit(train_trajectories_fit, external_trajectories_fit, fp_rate=fp_rate)

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
        
    def run_experiment(self) -> Tuple[float, float, float, float]:
        """
        Runs a basic MIA experiment, constructing data and train oracles and
        performing an MIA attack according to the given config. Returns accuracy,
        precision, and recall scores, along with the learned LRT threshold.
        """
        trainer_oracle, train_trajectories_fit, train_trajectories_nonfit, external_trajectories_fit, external_trajectories_nonfit = self._first_experiment_half()
        
        return self._second_experiment_half(trainer_oracle,
                                            train_trajectories_fit,
                                            train_trajectories_nonfit,
                                            external_trajectories_fit,
                                            external_trajectories_nonfit,
                                            self.fp_rate)      
    
    def test_fp_rates(self, fp_rate_list: List[float] = [0.05,0.1,0.15,0.2,0.25]) -> pd.DataFrame:
        trainer_oracle, train_trajectories_fit, train_trajectories_nonfit, external_trajectories_fit, external_trajectories_nonfit = self._first_experiment_half()
        
        accuracies = []
        precisions = []
        recalls = []
        etas = []
        
        for fp_rate in fp_rate_list:
            accuracy, precision, recall, eta = self._second_experiment_half(trainer_oracle,
                                            train_trajectories_fit,
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
    env = gym.make("Taxi-v3")
    T_max = 200
    verbose = 1
    seed = 1
    
    '''
    q_learner_config = QLearnerConfig(
        env=env,
        verbose=verbose
    )
    '''
    
    data_oracle_dqn_config = DQNConfig(
        env=env,
        policy="MlpPolicy",
        verbose=verbose,
        optimize_memory_usage=False,
        learning_starts=1000,
        exploration_fraction=0.2,
        buffer_size=10000
    )
    
    '''
    data_oracle_config = QLearnerDataOracleConfig(
        env=env,
        q_learner_config=q_learner_config,
        random_seed=seed,
        verbose=verbose
    )
    '''
    
    data_oracle_config=DQNDataOracleConfig(
        env=env,
        dqn_config = data_oracle_dqn_config,
        verbose=verbose
    )
    
    trainer_oracle_dqn_config = DQNConfig(
        env=env,
        policy="MlpPolicy",
        verbose=verbose,
        learning_starts=0,
        exploration_fraction=0,
        exploration_final_eps=0,
        #learning_starts=1000,
        #exploration_fraction=0.2,
        #buffer_size=10000,
        optimize_memory_usage=False
    )
    
    '''
    trainer_oracle_config = QLearnerTrainerOracleConfig(
        env=env,
        verbose=verbose,
        q_learner_config=q_learner_config
    )
    '''
    
    trainer_oracle_config = DQNTrainerOracleConfig(
        env=env,
        alpha=.0001,
        verbose=verbose,
        dqn_config=trainer_oracle_dqn_config
    )
    
    '''
    attacker_config = SARSAMIAConfig(
        alpha=.001,
        n_trajectories=1000,
        n_epochs = 20,
        T_max=T_max,
        seed=seed
    )
    '''
    
    experiment_config = ExperimentRunnerConfig(
        env=env,
        data_oracle_config=data_oracle_config,
        trainer_oracle_config=trainer_oracle_config,
        deep_data_oracle=True,
        deep_trainer_oracle=True,
        sarsa_attacker = False,
        sarsa_config=None,
        T_max=T_max,
        seed=seed,
        data_oracle_train_timesteps=1000000,
        n_trajectories=500,
        train_external_split=0.5,
        trainer_oracle_train_timesteps=1000000,
        fp_rate=0.05,
        mia_train_test_split=0.8
    )

    experiment_runner = ExperimentRunner(experiment_config)
    
    table = experiment_runner.test_fp_rates(fp_rate_list = [.05])
    table.to_csv("hyperparam_results_taxi_dqn_test.csv")
    
    print(table)
    
if __name__ == "__main__":
    main()