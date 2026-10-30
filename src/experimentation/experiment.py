import sys
import json
import itertools
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib

matplotlib.use(
    "Agg"
)  # non-interactive backend; the macOS one is unreliable in worker processes
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
    DeepTrainerOracleConfig,
    DeepDiscretizingTrainerOracleConfig,
    CustomFixedPolicyDataOracleConfig,
    QLearnerDataOracleConfig,
    QLearnerConfig,
    QLearnerTrainerOracleConfig,
    DQNDataOracleConfig,
    EnsemblingTrainerOracleConfig,
    DPAVIVITrainerOracleConfig,
    DPSGDTrainerOracleConfig
)

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))
from model import (
    MIAClassifier,
    ClusteringMIAClassifier,
    DeepTrainerOracle,
    DeepDiscretizingTrainerOracle,
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
    EnsemblingTrainerOracle,
    DPAVIVITrainerOracle,
    SepsisStateVectorObs,
    FeatureMappedDataOracle,
    DPSGDTrainerOracle,
)
from experimentation.roc_analysis import LOW_FPRS, tpr_at_fprs

# Remove verbose numpy warnings for glucose env
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="simglucose")
# Opacus computes per-sample gradients from module output gradients; inputs never require grad
warnings.filterwarnings("ignore", message="Full backward hook is firing when gradients are computed with respect to module outputs")


class ExperimentRunner:
    """
    Runs a membership inference attack experiment against a trained RL model.
    """

    def __init__(self, config: ExperimentRunnerConfig):
        self.config = config

    def _mean_reward_and_policy_heatmap(self):
        """
        Produces mean reward for current trainer oracle over 2000 trajectories,
        and makes a 1d heatmap of the policy
        """
        cfg = self.config
        eval_trajs = cfg.trainer_oracle.generate_trajectories(
            cfg.n_eval_trajectories, cfg.T_max
        )

        mean_reward = np.mean([sum(r for _, _, r, _ in traj) for traj in eval_trajs])
        if not cfg.save_plots:
            return mean_reward

        action_counts = Counter([trans[1] for traj in eval_trajs for trans in traj])
        action_counts = np.array(
            [action_counts[a] for a in all_actions(cfg.env.action_space)]
        )
        action_counts = action_counts.reshape(1, -1)
        action_counts = action_counts / action_counts.sum()
        fig, ax = plt.subplots(figsize=(8, 3.4))
        im = ax.imshow(
            action_counts, cmap="viridis", aspect="auto", interpolation="nearest"
        )
        ax.set_yticks([])
        ax.set_xlabel("Action")
        ax.set_title("Policy Heatmap")
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.3)
        cbar.set_label("Frequency (Normalized)")
        plt.savefig(
            f"data/plots/policy_heatmaps/{cfg.experiment_name}_heatmap.png"
        )
        plt.close("all")

        return mean_reward

    def _collect_trajectories(self):
        """
        Trains the oracle, generates trajectories, and splits them into
        member and non-member sets.

        Returns the two trajectory splits.
        """
        cfg = self.config
        n_train = round(cfg.n_trajectories * cfg.train_external_split)
        n_external = cfg.n_trajectories - n_train

        train_trajectories, external_trajectories = cfg.trainer_oracle.train(
            cfg.trainer_oracle_epochs, n_train, n_external, cfg.T_max, cfg.seed
        )

        # mean_reward = self._mean_reward_and_policy_heatmap()
        # print(f"Mean reward: {mean_reward}")

        return (
            train_trajectories,
            external_trajectories,
        )

    def _evaluate(
        self, train_trajs, external_trajs
    ) -> Tuple[float, float, float, float]:
        """
        Fits the MIA classifier and returns accuracy, precision, recall, roc/auc,
        mean member score, mean nonmember score, mean member q value, mean nonmember
        q value.
        """
        cfg = self.config
        all_trajs = train_trajs + external_trajs

        preds, scores, residuals, q_vals = cfg.mia_classifier.predict_memberships(
            all_trajs
        )

        train_preds = preds[: len(train_trajs)]
        external_preds = preds[len(train_trajs) :]

        member_scores = scores[: len(train_trajs)]
        member_residuals = np.concatenate(residuals[: len(train_trajs)])
        member_q_vals = np.concatenate(q_vals[: len(train_trajs)])
        nonmember_scores = scores[len(train_trajs) :]
        nonmember_residuals = np.concatenate(residuals[len(train_trajs) :])
        nonmember_q_vals = np.concatenate(q_vals[len(train_trajs) :])

        labels = np.concatenate(
            (np.ones(len(train_preds)), np.zeros(len(external_preds)))
        )

        if cfg.save_plots:
            MIAClassifier.plot_scores(member_scores, nonmember_scores, cfg.experiment_name)
            MIAClassifier.plot_residuals(
                member_residuals, nonmember_residuals, cfg.experiment_name
            )
        roc_auc = MIAClassifier.roc_auc(
            labels, np.concatenate((member_scores, nonmember_scores)), cfg.experiment_name,
            save_plot=cfg.save_plots,
        )

        # Raw per-trajectory results, saved by run_experiment for low-FPR / ROC analysis
        self._last_tprs = tuple(tpr_at_fprs(labels, scores))
        self._last_raw = {
            "scores": np.asarray(scores, dtype=np.float32),
            "residuals": np.concatenate(residuals).astype(np.float32),
            "q_vals": np.concatenate(q_vals).astype(np.float32),
        }

        return (
            accuracy_score(labels, preds),
            precision_score(labels, preds),
            recall_score(labels, preds),
            roc_auc,
            np.mean(member_scores),
            np.mean(nonmember_scores),
            np.mean(member_q_vals),
            np.mean(nonmember_q_vals),
        )

    def _save_raw_results(self, train_trajs, external_trajs, raw, result):
        """
        Saves one compressed .npz per run with per-trajectory scores and labels, so
        TPR at any FPR and aggregated ROC curves can be computed later (see roc_analysis). With save_transition_data, also saves the
        per-transition residuals, Q values, and (s, a, r, s') so results can be
        broken down by e.g. state-action coverage. Load with allow_pickle=True.
        """
        cfg = self.config
        all_trajs = train_trajs + external_trajs
        trainer_config = getattr(cfg.trainer_oracle, "config", None)
        meta = {
            "experiment_name": cfg.experiment_name,
            "seed": cfg.seed,
            "n_trajectories": cfg.n_trajectories,
            "n_members": len(train_trajs),
            "n_nonmembers": len(external_trajs),
            "trainer_oracle_epochs": cfg.trainer_oracle_epochs,
            "ensemble_k": getattr(trainer_config, "k", None),
            "low_fprs": list(LOW_FPRS),
            "reward": float(result[8]),
        }
        data = {
            "meta": np.array(json.dumps(meta)),
            "labels": np.concatenate(
                (np.ones(len(train_trajs)), np.zeros(len(external_trajs)))
            ).astype(np.int8),
            "scores": raw["scores"],
            "traj_lengths": np.array([len(traj) for traj in all_trajs], dtype=np.int32),
        }
        if cfg.save_transition_data:
            transitions = [t for traj in all_trajs for t in traj]
            for i, key in enumerate(("states", "actions", "rewards", "next_states")):
                column = [t[i] for t in transitions]
                try:
                    data[key] = np.asarray(
                        column, dtype=np.float64 if key == "rewards" else None
                    )
                except (ValueError, TypeError):
                    data[key] = np.array(column, dtype=object)
            data["residuals"] = raw["residuals"]
            data["q_vals"] = raw["q_vals"]

        Path(cfg.raw_results_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(cfg.raw_results_dir) / f"{cfg.experiment_name}.npz", **data
        )

    def run_experiment(self, trajectories: Tuple[List, List] = None) -> Tuple:
        """
        Runs a full MIA experiment and returns (accuracy, precision, recall, auc,
        member_mean_score, nonmember_mean_score, member_mean_q, nonmember_mean_q,
        trainer_mean_reward, *TPR at each of LOW_FPRS). If (member, non-member)
        trajectories are given, the trainer oracle is assumed to be already trained on
        the members and only the evaluation runs.
        """
        cfg = self.config
        if trajectories is None:
            train_trajs, external_trajs = self._collect_trajectories()
        else:
            train_trajs, external_trajs = trajectories
        result = self._evaluate(train_trajs, external_trajs)
        mean_reward = self._mean_reward_and_policy_heatmap()
        # Indices: 0-7 from _evaluate, 8 = mean reward, 9+ = TPR at each of LOW_FPRS
        result += (mean_reward,) + self._last_tprs

        if cfg.raw_results_dir:
            self._save_raw_results(train_trajs, external_trajs, self._last_raw, result)

        return result


def test_hyperparams(
    experiment_config: ExperimentRunnerConfig,
    n_trajectories_list: List[int] = [500],
    train_external_split_list: List[float] = [0.5],
    trainer_oracle_epochs_list: List[int] = [500],
    ensemble_k_list: List[int] = [1],
    ensemble_epsilon_list: List[float] = [1.0],
    csv_path: str = None,
) -> pd.DataFrame:
    """
    Produces a results table for every combination of the given hyperparameter
    lists. Does not mutate experiment_config. If csv_path is given, writes
    after each run so results are visible before all runs finish.
    """
    base_name = experiment_config.experiment_name
    rows = []

    for n_traj, split, epochs, ensemble_k, ensemble_eps in itertools.product(
        n_trajectories_list,
        train_external_split_list,
        trainer_oracle_epochs_list,
        ensemble_k_list,
        ensemble_epsilon_list,
    ):
        if isinstance(experiment_config.trainer_oracle, EnsemblingTrainerOracle):
            experiment_config.trainer_oracle.config.k = ensemble_k
            experiment_config.trainer_oracle.config.epsilon = ensemble_eps
        experiment_config.trainer_oracle.reset()
        config = replace(
            experiment_config,
            n_trajectories=n_traj,
            train_external_split=split,
            trainer_oracle_epochs=epochs,
            experiment_name=f"{base_name}_({ensemble_k},{n_traj},{split},{epochs})",
        )
        res = ExperimentRunner(config).run_experiment()
        rows.append(
            {
                "N. Trajectories": n_traj,
                "Train/External Traj Split": split,
                "Ensemble k": ensemble_k,
                "Ensemble epsilon": ensemble_eps,
                "Trainer Oracle Epochs": epochs,
                "Trainer Mean Reward": res[8],
                "Mean Member Score": res[4],
                "Mean Nonmember Score": res[5],
                "Mean Member Q value": res[6],
                "Mean Nonmember Q value": res[7],
                "Accuracy": res[0],
                "Precision": res[1],
                "Recall": res[2],
                "AUC": res[3],
                **{f"TPR@{fpr:g}FPR": tpr for fpr, tpr in zip(LOW_FPRS, res[9:])},
            }
        )
        if csv_path:
            pd.DataFrame(rows).to_csv(csv_path)

    return pd.DataFrame(rows)


def _result_row(n_traj, train_external_split, trainer_oracle_epochs, ensemble_k, ensemble_eps, res) -> dict:
    return {
        "N. Trajectories": n_traj,
        "Train/External Traj Split": train_external_split,
        "Ensemble k": ensemble_k,
        "Ensemble epsilon": ensemble_eps,
        "Trainer Oracle Epochs": trainer_oracle_epochs,
        "Trainer Mean Reward": res[8],
        "Mean Member Score": res[4],
        "Mean Nonmember Score": res[5],
        "Mean Member Q value": res[6],
        "Mean Nonmember Q value": res[7],
        "Accuracy": res[0],
        "Precision": res[1],
        "Recall": res[2],
        "AUC": res[3],
        **{f"TPR@{fpr:g}FPR": tpr for fpr, tpr in zip(LOW_FPRS, res[9:])},
    }


def _run_repeat(
    build_config: Callable[[int], ExperimentRunnerConfig],
    n_traj: int,
    train_external_split: float,
    trainer_oracle_epochs: int,
    ensemble_k: int,
    ensemble_eps_values: List,
    repeat: int,
    base_seed: int,
) -> List[dict]:
    """
    Runs one repeat of one hyperparameter setting in a worker process, building
    a fresh environment, trainer oracle, and MIA classifier. Returns result rows.
    With several epsilon values, the trainer oracle is trained once and then released
    and evaluated at each epsilon in turn (see TrainerOracle.release), giving one row
    and one raw results file per epsilon, named as if each had been its own run.
    """
    import random, torch

    torch.set_num_threads(1)

    seed = base_seed + repeat

    def seed_all():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Seed before building so network initialization is reproducible, and again
    # after, since some components (e.g. QLearnerDataOracle) reseed on init
    seed_all()
    experiment_config = build_config(ensemble_k, ensemble_eps_values[0])
    seed_all()

    def config_for(eps):
        return replace(
            experiment_config,
            n_trajectories=n_traj,
            train_external_split=train_external_split,
            trainer_oracle_epochs=trainer_oracle_epochs,
            seed=seed,
            experiment_name=f"{experiment_config.experiment_name}_({ensemble_k},{eps},{n_traj},{train_external_split},{trainer_oracle_epochs})_r{repeat}",
        )

    if len(ensemble_eps_values) == 1:
        res = ExperimentRunner(config_for(ensemble_eps_values[0])).run_experiment()
        return [_result_row(n_traj, train_external_split, trainer_oracle_epochs, ensemble_k, ensemble_eps_values[0], res)]

    trajectories = ExperimentRunner(config_for(ensemble_eps_values[0]))._collect_trajectories()
    rows = []
    for eps in ensemble_eps_values:
        experiment_config.trainer_oracle.release(eps)
        res = ExperimentRunner(config_for(eps)).run_experiment(trajectories=trajectories)
        rows.append(_result_row(n_traj, train_external_split, trainer_oracle_epochs, ensemble_k, eps, res))
    return rows


def test_hyperparams_parallel(
    build_config: Callable[[int], ExperimentRunnerConfig],
    n_workers: int,
    n_repeats: int,
    n_trajectories_list: List[int] = [500],
    train_external_split_list: List[float] = [0.5],
    trainer_oracle_epochs_list: List[int] = [500],
    ensemble_k_list: List[int] = [1],
    ensemble_epsilon_list: List[float] = [1.0],
    base_seed: int = 0,
    csv_path: str = None,
    share_training_across_epsilons: bool = False,
) -> pd.DataFrame:
    """
    Multiprocessing version of test_hyperparams. Runs n_repeats repeats of every
    hyperparameter combination across n_workers processes.

    With share_training_across_epsilons, each job trains once and evaluates every
    epsilon in ensemble_epsilon_list on that trained model (only for trainer oracles
    whose training does not depend on epsilon, i.e. that implement release). Training
    is seeded the same way as when each epsilon gets its own job, so the trained model
    is identical; only the noise draws for epsilons after the first differ.

    build_config must be a top-level (picklable) function that takes the ensemble k
    and constructs a fresh ExperimentRunnerConfig, including its env, trainer
    oracle, and MIA classifier.
    If csv_path is given, writes after each completed repeat.
    """
    eps_groups = [list(ensemble_epsilon_list)] if share_training_across_epsilons else [[eps] for eps in ensemble_epsilon_list]
    jobs = [
        (n_traj, split, epochs, k, eps_values, repeat)
        for n_traj, split, epochs, k, eps_values in itertools.product(
            n_trajectories_list,
            train_external_split_list,
            trainer_oracle_epochs_list,
            ensemble_k_list,
            eps_groups,
        )
        for repeat in range(n_repeats)
    ]
    rows = []

    with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [
            pool.submit(_run_repeat, build_config, *job, base_seed=base_seed)
            for job in jobs
        ]
        for i, f in tqdm(
            enumerate(as_completed(futures), start=1),
            desc=f"Running in parallel (/{len(jobs)})",
        ):
            rows.extend(f.result())
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
            train_states = np.array(
                [
                    np.atleast_1d(s).astype(float)
                    for traj in train_trajectories
                    for s, _, _, _ in traj
                ]
            )
            external_states = np.array(
                [
                    np.atleast_1d(s).astype(float)
                    for traj in external_trajectories
                    for s, _, _, _ in traj
                ]
            )
            if len(train_states) == 0 or len(external_states) == 0:
                return 0.0, 0
            tree = KDTree(train_states)  # enables quicker nearest calculations
            dists, _ = tree.query(external_states, k=1)
            return float(dists.sum()), len(external_states)

        else:

            def hashable(s):
                return tuple(s.tolist()) if isinstance(s, np.ndarray) else s

            member_counts = Counter(
                (hashable(s), a) for traj in train_trajectories for s, a, _, _ in traj
            )
            nonmember_counts = Counter(
                (hashable(s), a)
                for traj in external_trajectories
                for s, a, _, _ in traj
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


T_MAX = 50
DATA_ORACLE_TAU = 0.01
EXPERIMENT_NAME = "sepsis-cql-ensemble-clipped"
DATA_ORACLE_PATH = "models/sepsis_qlearner_data_oracle.pkl"


def _make_data_oracle(env: gym.Env) -> QLearnerDataOracle:
    q_learner = QLearner(
        QLearnerConfig(
            env=env,
            alpha=0.1,
            epsilon=10,
            buffer_size=1,
            discount_factor=1,
        )
    )
    return QLearnerDataOracle(
        QLearnerDataOracleConfig(
            env=env,
            decay_rate=0.9999991,
            learning_starts=10000,
            q_learner=q_learner,
            tau=DATA_ORACLE_TAU,
        )
    )


def build_config(ensemble_k: int, ensemble_eps: float) -> ExperimentRunnerConfig:
    """
    Builds a fresh experiment config in a worker process, with an ensemble of
    ensemble_k trainer oracles and a whole-table epsilon of ensemble_eps. The noise
    calibration (which depends on k and the number of training trajectories) is
    computed by the trainer oracle itself in train().
    The data oracle is loaded from disk so every repeat shares the same trained behavior policy.
    """
    env = SepsisStateVectorObs(gym.make("Sepsis/ICU-Sepsis-v2"))
    # The tabular behaviour policy acts on integer states; its trajectories are then mapped to feature vectors
    data_oracle = _make_data_oracle(gym.make("Sepsis/ICU-Sepsis-v2"))
    data_oracle.load(DATA_ORACLE_PATH)
    data_oracle = FeatureMappedDataOracle(data_oracle, env.encode)

    # Builds a new, independent sub-oracle on each call
    def make_inner_trainer_oracle():
        # return QLearnerTrainerOracle(
        #     QLearnerTrainerOracleConfig(
        #         env=env,
        #         alpha=0.001,
        #         discount_factor=1,
        #         data_oracle=data_oracle,
        #         q_learner=QLearner(
        #             QLearnerConfig(
        #                 env=env,
        #                 alpha=0.001,
        #                 buffer_size=1_000_000_000,
        #                 discount_factor=1,
        #                 unseen_training_val=0,
        #                 unseen_inference_val=0,
        #             )
        #         ),
        #     ),
        # )
        return DeepDiscretizingTrainerOracle(DeepDiscretizingTrainerOracleConfig(
           device="cpu",
           env=env,
           data_oracle=data_oracle,
           points=env.features,  # release a table over the 716 states' feature vectors
           cql_alpha=1.0,
           target_bounds=(0, 1),  # returns are in [0, 1]: discount 1, reward 1 on survival
           buffer_size=200_000,
           learning_rate=3e-4,
           batch_size=128,
           target_updates=30,
           discount_factor=1.0,
           gamma=1.0,
           net_arch=[128, 128],
        ))
        # return DPSGDTrainerOracle(DPSGDTrainerOracleConfig(
        #     device="cpu",
        #     env=env,
        #     data_oracle=data_oracle,
        #     cql_alpha=1.0,
        #     target_bounds=(0, 1),  # returns are in [0, 1]: discount 1, reward 1 on survival
        #     buffer_size=200_000,
        #     learning_rate=3e-4,
        #     batch_size=128,
        #     target_updates=30,
        #     discount_factor=1.0,
        #     gamma=1.0,
        #     hidden_sizes=(128,128),
        #     max_grad_norm=1.0,
            
        # ))

    # trainer_oracle = make_inner_trainer_oracle()
    trainer_oracle = EnsemblingTrainerOracle(
        EnsemblingTrainerOracleConfig(
            env=env,
            data_oracle=data_oracle,
            k=ensemble_k,
            trainer_factory=make_inner_trainer_oracle,
            q_bounds=(0, 1),
            epsilon=ensemble_eps,
            delta=1e-5,
        )
    )

    return ExperimentRunnerConfig(
        experiment_name=EXPERIMENT_NAME,
        env=env,
        trainer_oracle=trainer_oracle,
        mia_classifier=ClusteringMIAClassifier(trainer_oracle),
        T_max=T_MAX,
        seed=None,
        n_trajectories=500,
        train_external_split=0.5,
        trainer_oracle_epochs=500,
        mia_train_test_split=0.5,
        n_eval_trajectories=200,
        save_transition_data=False,
        save_plots=False,  # sweeps write one raw results file per run and no plots
    )


# DP-APVI experiment (Qiao & Wang 2023)
DPAPVI_EXPERIMENT_NAME = "cluster-sepsis-dpapvi"
DPAPVI_T_MAX = 50
DPAPVI_PENALTY_SCALE = 0.0   # the paper's Bernstein penalty (1.0) zeroes out Q on Sepsis

def build_dpapvi_config(ensemble_k: int, ensemble_eps: float) -> ExperimentRunnerConfig:
    """
    Builds a fresh experiment config in a worker process for DP-APVI: noisy counts of the
    member trajectories and pessimistic value iteration on the private model. ensemble_eps
    is the whole-release epsilon (None = exact counts, no privacy). ensemble_k is unused;
    it is accepted so the parallel runner can call every build function the same way.
    """
    env = gym.make("Sepsis/ICU-Sepsis-v2")
    data_oracle = _make_data_oracle(env)
    data_oracle.load(DATA_ORACLE_PATH)

    trainer_oracle = DPAVIVITrainerOracle(
        DPAVIVITrainerOracleConfig(
            env=env,
            discount_factor=1,  # the attack reads it from here
            data_oracle=data_oracle,
            epsilon=ensemble_eps,
            T_max=DPAPVI_T_MAX,
            value_bound=1.0,    # Sepsis's total reward is in [0, 1]
            penalty_scale=DPAPVI_PENALTY_SCALE,
        )
    )

    return ExperimentRunnerConfig(
        experiment_name=DPAPVI_EXPERIMENT_NAME,
        env=env,
        trainer_oracle=trainer_oracle,
        mia_classifier=ClusteringMIAClassifier(trainer_oracle),
        T_max=T_MAX,
        seed=None,
        n_trajectories=500,
        train_external_split=0.5,
        trainer_oracle_epochs=1,  # unused: value iteration runs T_max rounds
        mia_train_test_split=0.5,
        n_eval_trajectories=200,
        save_transition_data=False,
        save_plots=False,
    )

# DP-SGD experiment
DPSGD_EXPERIMENT_NAME = "sepsis-dpsgd"
DPSGD_T_MAX = 50
DPSGD_C = 1.0 # Max norm of grads

def build_dpsgd_config(ensemble_k: int, ensemble_eps: float) -> ExperimentRunnerConfig:
    """
    Builds a fresh experiment config in a worker process for DP-SGD: noisy gradients.
    """
    env = SepsisStateVectorObs(gym.make("Sepsis/ICU-Sepsis-v2"))
    # The tabular behaviour policy acts on integer states; its trajectories are then mapped to feature vectors
    data_oracle = _make_data_oracle(gym.make("Sepsis/ICU-Sepsis-v2"))
    data_oracle.load(DATA_ORACLE_PATH)
    data_oracle = FeatureMappedDataOracle(data_oracle, env.encode)

    trainer_oracle = DPSGDTrainerOracle(DPSGDTrainerOracleConfig(
        device="cpu",
        env=env,
        data_oracle=data_oracle,
        cql_alpha=1.0,
        target_bounds=(0, 1),  # returns are in [0, 1]: discount 1, reward 1 on survival
        learning_rate=3e-4,
        batch_size=128,
        target_updates=30,
        discount_factor=1.0,
        hidden_sizes=(64,64),
        max_grad_norm=DPSGD_C,
        epsilon=ensemble_eps,
        trajectory_loss="mean",
    ))

    return ExperimentRunnerConfig(
        experiment_name=DPSGD_EXPERIMENT_NAME,
        env=env,
        trainer_oracle=trainer_oracle,
        mia_classifier=ClusteringMIAClassifier(trainer_oracle),
        T_max=T_MAX,
        seed=None,
        n_trajectories=500,
        train_external_split=0.5,
        trainer_oracle_epochs=300,
        mia_train_test_split=0.5,
        n_eval_trajectories=200,
        save_transition_data=False,
        save_plots=False,
    )


# Which experiment main() runs
RUN = "dpsgd"

def main():
    env = gym.make("Sepsis/ICU-Sepsis-v2")
    data_oracle = _make_data_oracle(env)
    if Path(DATA_ORACLE_PATH).exists():
        data_oracle.load(DATA_ORACLE_PATH)
    else:
        data_oracle.train(5_000_000)
        data_oracle.save(DATA_ORACLE_PATH)

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

    eval_trajs = data_oracle.generate_trajectories(500, T_MAX)
    mean_reward = np.mean([sum(r for _, _, r, _ in traj) for traj in eval_trajs])
    print(f"Data oracle mean reward: {mean_reward:.4f}")
    states = [step[0] for traj in eval_trajs for step in traj]  # no set()
    ps = []
    for state in states:
        q_vals = data_oracle.q_vector(state)
        probs = np.exp((q_vals - np.max(q_vals)) / DATA_ORACLE_TAU)
        probs /= probs.sum()
        ps.append(np.max(probs))
    print(f"Expected greedy selection %: {np.mean(ps):.3f}")

    if RUN == "ensemble":
        csv_path = "data/tables/hyperparam_results_sepsis_cql_ensemble_clipped.csv"
        summary_path = "data/tables/summary_sepsis_cql_ensemble_clipped.csv"
        table = test_hyperparams_parallel(
            build_config=build_config,
            n_workers=6,
            n_repeats=50,
            n_trajectories_list=[100, 200, 500, 1000],
            trainer_oracle_epochs_list=[300],
            ensemble_k_list=[1, 5, 10, 20, 30, 50],
            ensemble_epsilon_list=[10, 300, 3000, 30000, 300000, 3000000, 30000000, 300000000, None],
            csv_path=csv_path,
            share_training_across_epsilons=True,  # train once per (N, k, repeat), release at every epsilon
        )
    elif RUN == "dpapvi":
        csv_path = "data/tables/hyperparam_results_cluster_sepsis_dpapvi.csv"
        summary_path = "data/tables/summary_cluster_sepsis_dpapvi.csv"
        table = test_hyperparams_parallel(
            build_config=build_dpapvi_config,
            n_workers=6,
            n_repeats=50,
            n_trajectories_list=[100, 200, 500, 1000, 4000, 20000],  # members + non-members; half are members
            trainer_oracle_epochs_list=[1],  # unused by DP-APVI
            ensemble_k_list=[1],  # unused by DP-APVI
            ensemble_epsilon_list=[None, 1, 10, 30, 100, 300, 1000, 3000, 10000],  # None = exact counts, no privacy
            csv_path=csv_path,
        )
    elif RUN == "dpsgd":
        csv_path = "data/tables/hyperparam_results_sepsis_dpsgd.csv"
        summary_path = "data/tables/summary_sepsis_dpsgd.csv"
        table = test_hyperparams_parallel(
            build_config=build_dpsgd_config,
            n_workers=4,
            n_repeats=50,
            n_trajectories_list=[1000],  # members + non-members; half are members
            trainer_oracle_epochs_list=[300],
            ensemble_k_list=[1],  # unused by DP-SGD
            ensemble_epsilon_list=[300000],
            csv_path=csv_path,
        )
    else:
        raise ValueError(f"Unknown RUN {RUN}")

    table.to_csv(csv_path)
    print(table)

    metric_cols = [
        "Trainer Mean Reward",
        "Accuracy",
        "Precision",
        "Recall",
        "AUC",
        *[f"TPR@{fpr:g}FPR" for fpr in LOW_FPRS],
    ]
    summary = (
        table.groupby(by=["N. Trajectories", "Ensemble k", "Ensemble epsilon"], dropna=False)[
            metric_cols
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.to_csv(summary_path)
    print("\nSummary (mean ± std):")
    print(summary)


if __name__ == "__main__":
    main()
