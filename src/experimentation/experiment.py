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
from stable_baselines3.common.atari_wrappers import AtariWrapper
import minigrid
from minigrid.wrappers import FullyObsWrapper
from collections import Counter

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
    DeepOnlineTrainerOracleConfig,
    DeepOfflineTrainerOracleConfig,
    CustomFixedPolicyDataOracleConfig
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
    DeepOnlineTrainerOracle,
    DeepOfflineTrainerOracle,
    CustomFixedPolicyDataOracle
)

from envs import (
    GridWorld
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
        self.experiment_name = config.experiment_name
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
        
        member_sa = set((s, a) for traj in train_trajectories 
                for s, a, _, _ in traj)

        nonmember_sa = set((s, a) for traj in external_trajectories 
                   for s, a, _, _ in traj)
        
        # Overlap
        # Count frequency of each (s,a) pair
        member_sa_counts = Counter(
            (s, a) for traj in train_trajectories 
            for s, a, _, _ in traj
        )

        nonmember_sa_counts = Counter(
            (s, a) for traj in external_trajectories 
            for s, a, _, _ in traj
        )

        # Total visits
        total_member = sum(member_sa_counts.values())
        total_nonmember = sum(nonmember_sa_counts.values())

        # Weighted overlap
        overlap_pairs = set(member_sa_counts.keys()) & set(nonmember_sa_counts.keys())

        # Fraction of non-member visits that are to overlapping states
        weighted_overlap = sum(nonmember_sa_counts[sa] for sa in overlap_pairs)
        print(f"Weighted overlap: {weighted_overlap/total_nonmember*100:.1f}%")
        print(f"Member unique (s,a) pairs: {len(member_sa_counts.keys())}")
        print(f"Nonmember unique (s,a) pairs: {len(nonmember_sa_counts.keys())}")
        
        #external_trajectories = self.trainer_oracle.data_oracle.generate_trajectories(n_external, self.T_max, self.seed)
        
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
                                fp_rate,
                                experiment_name):
        """
        Runs rest of experiment, training the MIA classifier for some fp rate.
        """
        #print(train_trajectories_fit)
        #print(external_trajectories_fit)
        
        self.mia_classifier.fit(train_trajectories_fit, external_trajectories_fit, fp_rate=fp_rate, experiment_name=experiment_name)

        # print(f"Learned eta: {mia_classifier.eta}")

        # Get predictions on trajs not used to fit
        train_predictions = self.mia_classifier.predict_memberships(train_trajectories_nonfit)
        external_predictions = self.mia_classifier.predict_memberships(external_trajectories_nonfit)
        
        member_scores = np.array([self.mia_classifier._traj_membership_score(traj) for traj in train_trajectories_nonfit])
        nonmember_scores = np.array([self.mia_classifier._traj_membership_score(traj) for traj in external_trajectories_nonfit])
        
        plt.figure()
        plt.hist(member_scores, alpha=0.5, label='Member', density=False)
        plt.hist(nonmember_scores, alpha=0.5, label='Non-member', density=False)
        plt.xlabel("Membership Score")
        plt.legend()
        plt.savefig(f"data/plots/test_member_nonmember_scores/{experiment_name}_score_hist.png")
        
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
                                            self.fp_rate,
                                            self.experiment_name)      
    

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
                                            fp_rate,
                                            f"{experiment_runner.experiment_name}_fp_rate_{fp_rate}")    
            
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
    
    original_experiment_name = experiment_config.experiment_name

    for n_trajectories in n_trajectories_list:
        for train_external_split in train_external_split_list:
            for trainer_oracle_train_timesteps in trainer_oracle_train_timesteps_list:
                for fp_rate in fp_rate_list:
                    experiment_config.trainer_oracle.reset()
                    experiment_config.n_trajectories = n_trajectories
                    experiment_config.train_external_split = train_external_split
                    experiment_config.trainer_oracle_train_timesteps = trainer_oracle_train_timesteps
                    experiment_config.fp_rate = fp_rate
                    experiment_config.experiment_name = f"{original_experiment_name}_({n_trajectories},{train_external_split},{trainer_oracle_train_timesteps},{fp_rate})"  
                    
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
    gym.register(
        id='GridWorld-v0',
        entry_point=GridWorld,
        kwargs={'size': 30, 'slip_prob': 0}
    )
    env = gym.make("Taxi-v3")
    T_max = 500
    verbose = 1
    seed = 1
    experiment_name = "taxi_qlearner_data_oracle_alpha_001"
    alpha = 0.001

    
    # Construct data oracle
    
    '''
    data_oracle = DQNDataOracle(
        DQNDataOracleConfig(
            env=env,
            dqn=DQN(policy="MlpPolicy",
                       env=env,
                       verbose=verbose,
                       learning_rate=0.00005,
                       learning_starts=1000,
                       exploration_fraction=0.7,
                       exploration_final_eps=0.05,
                       batch_size=64,
                       buffer_size=50000
            )
        )
    )
    data_oracle.train(learn_timesteps=1000000)
    
    
    
    data_oracle.dqn.save("models/acrobot_data_oracle_dqn")
    
    data_oracle = DQNDataOracle(
        DQNDataOracleConfig(
            env=env,
            dqn=DQN.load("models/acrobot_data_oracle_dqn", env=env)
        )
    )
    
    
    '''
    data_oracle = QLearnerDataOracle(
        QLearnerDataOracleConfig(
            env=env,
            q_learner=QLearner(QLearnerConfig(
                env=env,
                alpha=0.1,
                buffer_size=1,
                epsilon=1
            )),
            decay_rate=0.99995
        )
    )
    
    # Train for some timesteps
    data_oracle.train(500000)
    '''
    LOCS = [(0,0), (0,4), (4,0), (4,3)]  # R, G, Y, B

    def decode_state(state):
        dest = state % 4;        state //= 4
        pass_loc = state % 5;    state //= 5
        taxi_col = state % 5;    state //= 5
        taxi_row = state
        return taxi_row, taxi_col, pass_loc, dest
    
    WALLS = {
        # (row, col, direction) where direction is East(2) or West(3)
        (0, 1, 2),  # row 0, can't go East from col 1
        (0, 2, 3),  # row 0, can't go West from col 2
        (1, 1, 2),  # row 1, can't go East from col 1
        (1, 2, 3),  # row 1, can't go West from col 2
        (3, 0, 2),  # row 3, can't go East from col 0
        (3, 1, 3),  # row 3, can't go West from col 1
        (4, 0, 2),  # row 4, can't go East from col 0
        (4, 1, 3),  # row 4, can't go West from col 1
        (3, 2, 2),  # row 3, can't go East from col 2
        (3, 3, 3),  # row 3, can't go West from col 3
        (4, 2, 2),  # row 4, can't go East from col 2
        (4, 3, 3),  # row 4, can't go West from col 3
    }

    MOVES = {0: (1,0), 1: (-1,0), 2: (0,1), 3: (0,-1)}  # S,N,E,W

    def bfs(start, target):
        """Returns first action to take from start to reach target."""
        from collections import deque
        
        if start == target:
            return None
    
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            (row, col), path = queue.popleft()
            for action, (dr, dc) in MOVES.items():
                if (row, col, action) in WALLS:
                    continue
                nr, nc = row + dr, col + dc
                if not (0 <= nr < 5 and 0 <= nc < 5):
                    continue
                if (nr, nc) == target:
                    return path[0] if path else action
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [action]))

    def optimal_heuristic_policy(obs):
        taxi_row, taxi_col, pass_loc, dest = decode_state(obs)
        target = LOCS[pass_loc] if pass_loc < 4 else LOCS[dest]

        if (taxi_row, taxi_col) == target:
            return 4 if pass_loc < 4 else 5

        return bfs((taxi_row, taxi_col), target)
    
    data_oracle = CustomFixedPolicyDataOracle(CustomFixedPolicyDataOracleConfig(
        env=env,action_selector=optimal_heuristic_policy
    ))
    '''
    
    # Evaluate
    rewards = []
    for _ in range(100):
        obs, _ = env.reset()
        done = False
        total = 0
        t = 0
        while not done:
            action = data_oracle._select_action(
                data_oracle._encode_state(obs))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
            t += 1
            if t == T_max:
                break
        rewards.append(total)
    print(f"Mean reward: {np.mean(rewards):.4f}")
    
    
    trainer_oracle = QLearnerTrainerOracle(
        QLearnerTrainerOracleConfig(
            env=env,
            data_oracle=data_oracle,
            q_learner=QLearner(QLearnerConfig(
                env=env,
                alpha=alpha,
                buffer_size=1000000000
            )),
            alpha=alpha
        )
    )
    #
    
    '''
    # Use deep trainer oracle
    
    trainer_oracle = DeepOfflineTrainerOracle(
        DeepOfflineTrainerOracleConfig(
            env=env,
            alpha=alpha,
            dqn=DQN(policy="MlpPolicy",
                       env=env,
                       verbose=verbose,
                       learning_rate=0.0005,
                       learning_starts=0,
                       exploration_fraction=0,
                       exploration_final_eps=0,
                       batch_size=32,
                       buffer_size=10000000),
            data_oracle=data_oracle
        )
    )
    '''
    
    mia_classifier = MIAClassifier(
        trainer_oracle
    )
    
    experiment_config = ExperimentRunnerConfig(
        experiment_name=experiment_name,
        env=env,
        trainer_oracle=trainer_oracle,
        mia_classifier=mia_classifier,
        T_max=T_max,
        seed=seed,
        n_trajectories=250,
        train_external_split=0.5,
        trainer_oracle_train_timesteps=200000,
        fp_rate=0.25,
        mia_train_test_split=0.5
    )
    
    
    table = test_hyperparams(experiment_config, n_trajectories_list=[125], trainer_oracle_train_timesteps_list=[7000000]*20, fp_rate_list=[0.25])
    
    #experiment_runner = ExperimentRunner(experiment_config)
    #table = test_fp_rates(
    #    experiment_runner,
    #    fp_rate_list=[0.1,0.2,0.3,0.4]
    #)
    
    table.to_csv("data/tables/hyperparam_results_gridworld.csv")
    
    print(table)
    
    # Plot metrics vs fp rates
    plt.figure()
    plt.plot(table["FP Rate"], table["Accuracy"], label='Accuracy')
    plt.plot(table["FP Rate"], table["Precision"], label='Precision')
    plt.plot(table["FP Rate"], table["Recall"], label='Recall')
    plt.xlabel("FP Rate")
    plt.savefig(f"data/plots/fp_rates/{experiment_name}_fp_rates_plot.png")
    
if __name__ == "__main__":
    main()