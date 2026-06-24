import sys
import itertools
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from collections import Counter
import pandas as pd
import icu_sepsis
from minigrid.wrappers import FlatObsWrapper

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
from envs import GridWorld
from policies import lunar_lander_policy, make_expert_policy, BabyAIBotEnv, make_minigrid_bot_policy


class ExperimentRunner:
    """
    Runs a membership inference attack experiment against a trained RL model.
    """

    def __init__(self, config: ExperimentRunnerConfig):
        self.config = config

    def _collect_trajectories(self):
        """
        Trains the oracle, generates trajectories, and splits them into
        fit/eval sets for both member and non-member populations.
        """
        cfg = self.config
        n_train = round(cfg.n_trajectories * cfg.train_external_split)
        n_external = cfg.n_trajectories - n_train

        train_trajectories, external_trajectories = cfg.trainer_oracle.train(
            cfg.trainer_oracle_train_timesteps, n_train, n_external, cfg.T_max, cfg.seed
        )

        # Diagnostic: (s,a) overlap between member and non-member sets
        member_counts = Counter(
            (s, a) for traj in train_trajectories for s, a, _, _ in traj
        )
        nonmember_counts = Counter(
            (s, a) for traj in external_trajectories for s, a, _, _ in traj
        )
        overlap_pairs = member_counts.keys() & nonmember_counts.keys()
        total_nonmember = sum(nonmember_counts.values())
        weighted_overlap = sum(nonmember_counts[sa] for sa in overlap_pairs)
        print(f"Weighted overlap: {weighted_overlap / total_nonmember * 100:.1f}%")
        print(f"Member unique (s,a) pairs: {len(member_counts)}")
        print(f"Nonmember unique (s,a) pairs: {len(nonmember_counts)}")

        train_split = round(cfg.mia_train_test_split * n_train)
        external_split = round(cfg.mia_train_test_split * n_external)

        return (
            train_trajectories[:train_split],
            train_trajectories[train_split:],
            external_trajectories[:external_split],
            external_trajectories[external_split:],
        )

    def _evaluate(self,
                  train_fit, train_eval,
                  external_fit, external_eval,
                  fp_rate: float,
                  experiment_name: str) -> Tuple[float, float, float, float]:
        """
        Fits the MIA classifier and returns accuracy, precision, recall, and eta.
        """
        cfg = self.config
        cfg.mia_classifier.fit(train_fit, external_fit, fp_rate=fp_rate, experiment_name=experiment_name)

        train_preds = cfg.mia_classifier.predict_memberships(train_eval)
        external_preds = cfg.mia_classifier.predict_memberships(external_eval)

        member_scores = np.array([cfg.mia_classifier._traj_membership_score(t) for t in train_eval])
        nonmember_scores = np.array([cfg.mia_classifier._traj_membership_score(t) for t in external_eval])

        plt.figure()
        plt.hist(member_scores, alpha=0.5, label='Member')
        plt.hist(nonmember_scores, alpha=0.5, label='Non-member')
        plt.xlabel("Membership Score")
        plt.legend()
        plt.savefig(f"data/plots/test_member_nonmember_scores/{experiment_name}_score_hist.png")
        plt.close()

        preds = np.concatenate((train_preds, external_preds))
        labels = np.concatenate((np.ones(len(train_preds)), np.zeros(len(external_preds))))

        return (
            accuracy_score(labels, preds),
            precision_score(labels, preds),
            recall_score(labels, preds),
            cfg.mia_classifier.eta,
        )

    def run_experiment(self) -> Tuple[float, float, float, float]:
        """
        Runs a full MIA experiment and returns (accuracy, precision, recall, eta).
        """
        train_fit, train_eval, external_fit, external_eval = self._collect_trajectories()
        return self._evaluate(
            train_fit, train_eval, external_fit, external_eval,
            self.config.fp_rate, self.config.experiment_name
        )

    def test_fp_rates(self, fp_rate_list: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25]) -> pd.DataFrame:
        """
        Runs the MIA classifier at each fp_rate without re-training the oracle.
        """
        train_fit, train_eval, external_fit, external_eval = self._collect_trajectories()
        rows = []
        for fp_rate in fp_rate_list:
            name = f"{self.config.experiment_name}_fp_rate_{fp_rate}"
            accuracy, precision, recall, eta = self._evaluate(
                train_fit, train_eval, external_fit, external_eval, fp_rate, name
            )
            rows.append({"FP Rate": fp_rate, "Accuracy": accuracy,
                         "Precision": precision, "Recall": recall, "Eta": eta})
        return pd.DataFrame(rows)


def test_hyperparams(
    experiment_config: ExperimentRunnerConfig,
    n_trajectories_list: List[int] = [500],
    train_external_split_list: List[float] = [0.5],
    trainer_oracle_train_timesteps_list: List[int] = [1000000],
    fp_rate_list: List[float] = [0.05],
) -> pd.DataFrame:
    """
    Produces a results table for every combination of the given hyperparameter
    lists. Does not mutate experiment_config.
    """
    base_name = experiment_config.experiment_name
    rows = []

    for n_traj, split, steps, fp_rate in itertools.product(
        n_trajectories_list,
        train_external_split_list,
        trainer_oracle_train_timesteps_list,
        fp_rate_list,
    ):
        experiment_config.trainer_oracle.reset()
        config = replace(
            experiment_config,
            n_trajectories=n_traj,
            train_external_split=split,
            trainer_oracle_train_timesteps=steps,
            fp_rate=fp_rate,
            experiment_name=f"{base_name}_({n_traj},{split},{steps},{fp_rate})",
        )
        accuracy, precision, recall, eta = ExperimentRunner(config).run_experiment()
        rows.append({
            "N. Trajectories": n_traj,
            "Train/External Traj Split": split,
            "Trainer Oracle Timesteps": steps,
            "FP Rate": fp_rate,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Eta": eta,
        })

    return pd.DataFrame(rows)


def main():
    env = BabyAIBotEnv(FlatObsWrapper(gym.make("BabyAI-GoToObj-v0")))
    obs, _ = env.reset()
    print(f"Obs length: {len(obs)}")

    T_max = 200
    verbose = 1
    seed = None
    experiment_name = "minigrid"
    alpha = 0.001

    policy = make_minigrid_bot_policy(env)
    data_oracle = CustomFixedPolicyDataOracle(CustomFixedPolicyDataOracleConfig(
        env=env, action_selector=policy
    ))

    # Sanity-check the data oracle's policy performance
    eval_trajs = data_oracle.generate_trajectories(50, T_max)
    mean_reward = np.mean([sum(r for _, _, r, _ in traj) for traj in eval_trajs])
    print(f"Mean reward: {mean_reward:.4f}")

    trainer_oracle = DeepOfflineTrainerOracle(
        DeepOfflineTrainerOracleConfig(
            env=env,
            alpha=alpha,
            learning_rate=0.0005,
            learning_starts=0,
            exploration_fraction=0,
            exploration_final_eps=0,
            batch_size=32,
            buffer_size=10000000,
            data_oracle=data_oracle
        )
    )

    mia_classifier = MIAClassifier(trainer_oracle)

    experiment_config = ExperimentRunnerConfig(
        experiment_name=experiment_name,
        env=env,
        trainer_oracle=trainer_oracle,
        mia_classifier=mia_classifier,
        T_max=T_max,
        seed=seed,
        n_trajectories=125,
        train_external_split=0.5,
        trainer_oracle_train_timesteps=1000000,
        fp_rate=0.25,
        mia_train_test_split=0.5
    )

    table = test_hyperparams(
        experiment_config,
        n_trajectories_list=[125],
        trainer_oracle_train_timesteps_list=[2000000,5000000,10000000,20000000],
        fp_rate_list=[0.25]
    )

    table.to_csv("data/tables/hyperparam_results_minigrid.csv")
    print(table)

    setting_cols = ["N. Trajectories", "Train/External Traj Split", "Trainer Oracle Timesteps", "FP Rate"]
    metric_cols = ["Accuracy", "Precision", "Recall"]
    means = table.groupby(setting_cols)[metric_cols].mean().reset_index()
    print("\nMeans:")
    print(means)

    plt.figure()
    plt.plot(table["FP Rate"], table["Accuracy"], label='Accuracy')
    plt.plot(table["FP Rate"], table["Precision"], label='Precision')
    plt.plot(table["FP Rate"], table["Recall"], label='Recall')
    plt.xlabel("FP Rate")
    plt.legend()
    plt.savefig(f"data/plots/fp_rates/{experiment_name}_fp_rates_plot.png")


if __name__ == "__main__":
    main()
