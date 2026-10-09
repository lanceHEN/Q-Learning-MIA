import sys
import itertools
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.spatial import KDTree
from collections import Counter
import pandas as pd
import icu_sepsis  # registers Sepsis/ICU-Sepsis-v2 with gymnasium

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
from config import (
    ExperimentRunnerConfig,
    DeepOfflineTrainerOracleConfig,
    CustomFixedPolicyDataOracleConfig,
    QLearnerDataOracleConfig,
    QLearnerConfig,
    QLearnerTrainerOracleConfig,
    DQNDataOracleConfig,
)

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))
from model import (
    MIAClassifier,
    ClusteringMIAClassifier,
    DeepOfflineTrainerOracle,
    CustomFixedPolicyDataOracle,
    QLearnerDataOracle,
    QLearner,
    QLearnerTrainerOracle,
    TupleObsToBox,
    DQNDataOracle,
    DataOracle,
    HIVTreatmentEnv,
    ChemotherapyEnv,
    all_actions,
)

# Remove verbose numpy warnings for glucose env
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="simglucose")

class ExperimentRunner:
    """
    Runs a membership inference attack experiment against a trained RL model.
    """

    def __init__(self, config: ExperimentRunnerConfig):
        self.config = config  
        
    def _mean_reward_and_policy_heatmap(self, std):
        """
        Produces mean reward for current trainer oracle over 2000 trajectories,
        and makes a 1d heatmap of the policy
        """
        cfg = self.config
        eval_trajs = cfg.trainer_oracle.generate_trajectories(cfg.n_eval_trajectories, cfg.T_max)
        
        action_counts = Counter([trans[1] for traj in eval_trajs for trans in traj])
        action_counts = np.array([action_counts[a] for a in all_actions(cfg.env.action_space)])
        action_counts = action_counts.reshape(1, -1)
        action_counts = action_counts / action_counts.sum()
        fig, ax = plt.subplots(figsize=(8, 3.4))
        im = ax.imshow(action_counts, cmap="viridis", aspect="auto", interpolation="nearest")
        ax.set_yticks([])
        ax.set_xlabel("Action")
        ax.set_title("Policy Heatmap")
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.3)
        cbar.set_label("Frequency (Normalized)")
        plt.savefig(f"data/plots/policy_heatmaps/{cfg.experiment_name}_std_{std}_heatmap.png")
        plt.close('all')
        
        return np.mean([sum(r for _, _, r, _ in traj) for traj in eval_trajs])

    def _collect_trajectories(self):
        """
        Trains the oracle, generates trajectories, and splits them into
        member and non-member sets.

        Returns the two trajectory splits.
        """
        cfg = self.config
        cfg.trainer_oracle.perturb = False
        n_train = round(cfg.n_trajectories * cfg.train_external_split)
        n_external = cfg.n_trajectories - n_train

        train_trajectories, external_trajectories = cfg.trainer_oracle.train(
            cfg.trainer_oracle_train_timesteps, n_train, n_external, cfg.T_max, cfg.seed
        )
        
        #mean_reward = self._mean_reward_and_policy_heatmap(std=0)
        #print(f"Mean reward: {mean_reward}")

        return (
            train_trajectories,
            external_trajectories,
        )

    def _evaluate(self,
                  train_trajs,
                  external_trajs,
                  experiment_name: str) -> Tuple[float, float, float, float]:
        """
        Fits the MIA classifier and returns accuracy, precision, recall, and roc/auc.
        """
        cfg = self.config
        cfg.trainer_oracle.perturb = True
        
        all_trajs = train_trajs + external_trajs
        
        preds, scores, residuals = cfg.mia_classifier.predict_memberships(all_trajs)

        train_preds = preds[:len(train_trajs)]
        external_preds = preds[len(train_trajs):]
            
        member_scores = scores[:len(train_trajs)]
        member_residuals = np.concatenate(residuals[:len(train_trajs)])
        nonmember_scores = scores[len(train_trajs):]
        nonmember_residuals = np.concatenate(residuals[len(train_trajs):])

        labels = np.concatenate((np.ones(len(train_preds)), np.zeros(len(external_preds))))

        MIAClassifier.plot_scores(member_scores, nonmember_scores, cfg.experiment_name)
        MIAClassifier.plot_residuals(member_residuals, nonmember_residuals, cfg.experiment_name)
        roc_auc = MIAClassifier.roc_auc(labels, np.concatenate((member_scores, nonmember_scores)), experiment_name)

        return (
            accuracy_score(labels, preds),
            precision_score(labels, preds),
            recall_score(labels, preds),
            roc_auc,
        )        

    def run_experiment(self) -> Tuple[float, float, float, float, float]:
        """
        Runs a full MIA experiment and returns a list of
        (accuracy, precision, recall, auc, and trainer_mean_reward, for each
        DP std level.
        """
        cfg = self.config
        train_trajs, external_trajs = self._collect_trajectories()
        results = []
        for std in cfg.output_perturbation_stds:
            cfg.trainer_oracle.perturbation_std = std
            res = self._evaluate(
                train_trajs, external_trajs, f"{cfg.experiment_name}_std_{std}"
            )
            mean_reward = self._mean_reward_and_policy_heatmap(std=std)
            res += (mean_reward,)
            results.append(res)
            
        # Mean reward vs std
        plt.figure()
        plt.plot(cfg.output_perturbation_stds, [res[4] for res in results], marker="o")
        plt.xlabel("Output Perturbation Std")
        plt.ylabel("Mean Rewards")
        plt.savefig(f"data/plots/rewards_vs_perturbations/{cfg.experiment_name}_plot.png")
        plt.close('all')
        
        # AUC vs std
        plt.figure()
        plt.plot(cfg.output_perturbation_stds, [res[3] for res in results], marker="o")
        plt.xlabel("Output Perturbation Std")
        plt.ylabel("AUC")
        plt.savefig(f"data/plots/auc_vs_perturbations/{cfg.experiment_name}_plot.png")
        plt.close('all')
            
        return results

def test_hyperparams(
    experiment_config: ExperimentRunnerConfig,
    n_trajectories_list: List[int] = [500],
    train_external_split_list: List[float] = [0.5],
    trainer_oracle_train_timesteps_list: List[int] = [1000000],
    csv_path: str = None,
) -> pd.DataFrame:
    """
    Produces a results table for every combination of the given hyperparameter
    lists. Does not mutate experiment_config. If csv_path is given, writes
    after each run so results are visible before all runs finish.
    """
    base_name = experiment_config.experiment_name
    rows = []

    for n_traj, split, steps in itertools.product(
        n_trajectories_list,
        train_external_split_list,
        trainer_oracle_train_timesteps_list,
    ):
        experiment_config.trainer_oracle.reset()
        config = replace(
            experiment_config,
            n_trajectories=n_traj,
            train_external_split=split,
            trainer_oracle_train_timesteps=steps,
            experiment_name=f"{base_name}_({n_traj},{split},{steps})",
        )
        results = ExperimentRunner(config).run_experiment()
        rs = [{
            "N. Trajectories": n_traj,
            "Train/External Traj Split": split,
            "Output Perturbation std": experiment_config.output_perturbation_stds[i],
            "Trainer Oracle Timesteps": steps,
            "Trainer Mean Reward": res[4],
            "Accuracy": res[0],
            "Precision": res[1],
            "Recall": res[2],
            "AUC": res[3],
        } for i, res in enumerate(results)]
        rows.extend(rs)
        if csv_path:
            pd.DataFrame(rows).to_csv(csv_path)

    return pd.DataFrame(rows)

def overlap_metrics(
    env: gym.Env,
    data_oracle: DataOracle,
    T_max: int,
    train_external_split: float = 0.5,
    n_trajectories_list: List[int] = [500],
    csv_path: str = None,
) -> pd.DataFrame:
    """
    Measures (s,a) overlap between member and non-member trajectories for
    each n in n_trajectories_list, without running the full MIA pipeline.

    Discrete observation space:
        Returns (weighted_overlap, total_nonmember) where weighted_overlap
        counts non-member transitions whose exact (s,a) pair appears in the
        member set. Overlap fraction = weighted_overlap / total_nonmember.

    Continuous observation space (Box):
        Returns (sum_of_nearest_distances, total_nonmember) where each
        non-member state contributes its Euclidean distance to the nearest
        member state. Mean nearest-member distance = sum / total_nonmember;
        lower values indicate higher coverage (more overlap).

    In both cases raw counts are stored so that results can be summed across
    repeated runs before computing the final metric, avoiding mean-of-means
    aggregation errors.
    """
    continuous = isinstance(env.observation_space, gym.spaces.Box)

    def individual_overlap_metrics(n_trajectories: int):
        n_train = round(n_trajectories * train_external_split)
        n_external = n_trajectories - n_train

        train_trajectories = data_oracle.generate_trajectories(n_train, T_max)
        external_trajectories = data_oracle.generate_trajectories(n_external, T_max)

        if continuous:
            train_states = np.array([
                np.atleast_1d(s).astype(float)
                for traj in train_trajectories for s, _, _, _ in traj
            ])
            external_states = np.array([
                np.atleast_1d(s).astype(float)
                for traj in external_trajectories for s, _, _, _ in traj
            ])
            if len(train_states) == 0 or len(external_states) == 0:
                return 0.0, 0
            tree = KDTree(train_states) # enables quicker nearest calculations
            dists, _ = tree.query(external_states, k=1)
            return float(dists.sum()), len(external_states)

        else:
            def hashable(s):
                return tuple(s.tolist()) if isinstance(s, np.ndarray) else s

            member_counts = Counter(
                (hashable(s), a) for traj in train_trajectories for s, a, _, _ in traj
            )
            nonmember_counts = Counter(
                (hashable(s), a) for traj in external_trajectories for s, a, _, _ in traj
            )
            overlap_pairs = member_counts.keys() & nonmember_counts.keys()
            total_nonmember = sum(nonmember_counts.values())
            weighted_overlap = sum(nonmember_counts[sa] for sa in overlap_pairs)
            return weighted_overlap, total_nonmember

    rows = []
    for n_traj in n_trajectories_list:
        m1, nonmember_pairs = individual_overlap_metrics(n_traj)
        row = {
            "N. Trajectories": n_traj,
            ("Nearest Distance Sum" if continuous else "Nonmember Overlap Pairs"): m1,
            "Nonmember Pairs": nonmember_pairs,
        }
        rows.append(row)
        if csv_path:
            pd.DataFrame(rows).to_csv(csv_path)

    return pd.DataFrame(rows)



def main():
    T_max = 100
    tau = 0.01 # .005 .91, 
    env = gym.make("Sepsis/ICU-Sepsis-v2")
    seed = None
    experiment_name = "cluster-sepsis-tabular-dp-nondeterministic"

    q_learner = QLearner(QLearnerConfig(
        env=env,
        alpha=0.1,
        epsilon=1,
        buffer_size=1,
        discount_factor=1,
    ))
    data_oracle = QLearnerDataOracle(QLearnerDataOracleConfig(
        env=env,
        decay_rate=0.9999991,
        learning_starts=10000,
        q_learner=q_learner,
        tau=tau,
    ))
    data_oracle.train(5_000_000)

    # data_oracle = DQNDataOracle(DQNDataOracleConfig(
    #    env=env,
    #    learning_rate=3e-4,
    #    learning_starts=1_000,
    #    exploration_fraction=0.4,
    #    exploration_final_eps=0.01,
    #    batch_size=64,
    #    buffer_size=100_000,
    #    device="cpu",
    #    policy_kwargs={"net_arch": [64, 64]},
    #    target_update_interval=1000,
    #    gamma=1,
    # ))
    # data_oracle.train(500_000)
    # data_oracle.save("models/sepsis_dqn")
    

    eval_trajs = data_oracle.generate_trajectories(500, T_max)
    mean_reward = np.mean([sum(r for _, _, r, _ in traj) for traj in eval_trajs])
    print(f"Data oracle mean reward: {mean_reward:.4f}")
    states = [step[0] for traj in eval_trajs for step in traj]     # no set()
    ps = []
    for state in states:
        q_vals = data_oracle.q_vector(state)
        probs = np.exp((q_vals - np.max(q_vals)) / tau)
        probs /= probs.sum()
        ps.append(np.max(probs))
    print(f"Expected greedy selection %: {np.mean(ps):.3f}")
    
    trainer_oracle = QLearnerTrainerOracle(QLearnerTrainerOracleConfig(
        env=env,
        alpha=0.001,
        discount_factor=1,
        data_oracle=data_oracle,
        q_learner=QLearner(QLearnerConfig(
            env=env,
            alpha=0.001,
            buffer_size=1_000_000_000,
            discount_factor=1,
        )),
    ))

    # trainer_oracle = DeepOfflineTrainerOracle(DeepOfflineTrainerOracleConfig(
    #    device="cpu",
    #    env=env,
    #    data_oracle=data_oracle,
    #    cql_alpha=2.0,
    #    buffer_size=10_000_000,
    #    gradient_steps=500_000,
    #    learning_rate=1e-3,
    #    batch_size=128,
    #    tau=0.01,
    # ))

    mia_classifier = ClusteringMIAClassifier(trainer_oracle)

    experiment_config = ExperimentRunnerConfig(
        experiment_name=experiment_name,
        env=env,
        trainer_oracle=trainer_oracle,
        mia_classifier=mia_classifier,
        T_max=T_max,
        seed=seed,
        n_trajectories=500,
        train_external_split=0.5,
        trainer_oracle_train_timesteps=0,
        mia_train_test_split=0.5,
        output_perturbation_stds=[0, 0.1, 0.5, 1.0, 5.0],
        n_eval_trajectories=200,
    )

    csv_path = "data/tables/hyperparam_results_cluster_sepsis_tabular_dp_nondeterministic.csv"
    table = test_hyperparams(
        experiment_config,
        n_trajectories_list=[100, 200, 500, 1000],
        trainer_oracle_train_timesteps_list=[10_000_000]*50,
        csv_path=csv_path,
    )

    table.to_csv(csv_path)
    print(table)

    metric_cols = ["Trainer Mean Reward", "Accuracy", "Precision", "Recall", "AUC"]
    summary = table.groupby(by=["N. Trajectories", "Output Perturbation std"])[metric_cols].agg(["mean", "std"]).round(4)
    summary.to_csv("data/tables/summary_cluster_sepsis_tabular_dp_nondeterministic.csv")
    print("\nSummary (mean ± std):")
    print(summary)

if __name__ == "__main__":
    main()
