"""
Runs MIA experiments: ExperimentRunner trains a trainer oracle and attacks it once, and
run_sweep repeats that over a grid of settings in parallel. Nothing here is specific to an
environment or mitigation; those are built by the functions in setups.py.
"""
import itertools
import json
import multiprocessing as mp
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; the macOS one is unreliable in worker processes
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm

from config import ExperimentRunnerConfig
from src.model import MIAClassifier, all_actions
from .roc_analysis import LOW_FPRS, tpr_at_fprs

# Verbose numpy deprecation warnings from simglucose
warnings.filterwarnings("ignore", category=DeprecationWarning, module="simglucose")
# Opacus computes per-sample gradients from module output gradients; inputs never require grad
warnings.filterwarnings("ignore", message="Full backward hook is firing when gradients are computed with respect to module outputs")

METRIC_COLUMNS = [
    "Trainer Mean Reward",
    "Accuracy",
    "Precision",
    "Recall",
    "AUC",
    *[f"TPR@{fpr:g}FPR" for fpr in LOW_FPRS],
]


class ExperimentRunner:
    """
    Runs a membership inference attack experiment against a trained RL model.
    """

    def __init__(self, config: ExperimentRunnerConfig):
        self.config = config

    def _mean_reward_and_policy_heatmap(self):
        """
        Mean return of the trainer oracle's policy over n_eval_trajectories episodes. With
        save_plots, also saves a 1-D heatmap of how often it takes each action.
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

    def collect_trajectories(self) -> Tuple[List, List]:
        """
        Trains the trainer oracle and returns its (member, non-member) trajectories.
        """
        cfg = self.config
        n_train = round(cfg.n_trajectories * cfg.train_external_split)
        n_external = cfg.n_trajectories - n_train
        return cfg.trainer_oracle.train(
            cfg.trainer_oracle_epochs, n_train, n_external, cfg.T_max, cfg.seed
        )

    def _evaluate(self, train_trajs, external_trajs) -> Tuple:
        """
        Runs the attack and returns accuracy, precision, recall, AUC, mean member score,
        mean non-member score, mean member Q value, and mean non-member Q value.
        """
        cfg = self.config
        all_trajs = train_trajs + external_trajs

        preds, scores, residuals, q_vals, targets = cfg.mia_classifier.predict_memberships(
            all_trajs
        )

        n_members = len(train_trajs)
        member_scores = scores[:n_members]
        nonmember_scores = scores[n_members:]
        member_q_vals = np.concatenate(q_vals[:n_members])
        nonmember_q_vals = np.concatenate(q_vals[n_members:])

        labels = np.concatenate(
            (np.ones(n_members), np.zeros(len(external_trajs)))
        )

        if cfg.save_plots:
            MIAClassifier.plot_scores(member_scores, nonmember_scores, cfg.experiment_name)
            MIAClassifier.plot_residuals(
                np.concatenate(residuals[:n_members]),
                np.concatenate(residuals[n_members:]),
                cfg.experiment_name,
            )
            MIAClassifier.plot_targets(
                np.concatenate(targets[:n_members]),
                np.concatenate(targets[n_members:]),
                cfg.experiment_name,
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
        Saves one compressed .npz per run with per-trajectory scores and labels, so TPR at
        any FPR and aggregated ROC curves can be computed later (see roc_analysis). With
        save_transition_data, also saves the per-transition residuals, Q values, and
        (s, a, r, s') so results can be broken down by e.g. state-action coverage. Load
        with allow_pickle=True.
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
            train_trajs, external_trajs = self.collect_trajectories()
        else:
            train_trajs, external_trajs = trajectories
        result = self._evaluate(train_trajs, external_trajs)
        mean_reward = self._mean_reward_and_policy_heatmap()
        # Indices: 0-7 from _evaluate, 8 = mean reward, 9+ = TPR at each of LOW_FPRS
        result += (mean_reward,) + self._last_tprs

        if cfg.raw_results_dir:
            self._save_raw_results(train_trajs, external_trajs, self._last_raw, result)

        return result


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
    build_config: Callable[[int, float], ExperimentRunnerConfig],
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
    and evaluated at each epsilon in turn (see EnsemblingTrainerOracle.release), giving
    one row and one raw results file per epsilon, named as if each had been its own run.
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

    trajectories = ExperimentRunner(config_for(ensemble_eps_values[0])).collect_trajectories()
    rows = []
    for eps in ensemble_eps_values:
        experiment_config.trainer_oracle.release(eps)
        res = ExperimentRunner(config_for(eps)).run_experiment(trajectories=trajectories)
        rows.append(_result_row(n_traj, train_external_split, trainer_oracle_epochs, ensemble_k, eps, res))
    return rows


def run_sweep(
    build_config: Callable[[int, float], ExperimentRunnerConfig],
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
    Produces a results table for every combination of the given hyperparameter lists,
    running n_repeats repeats of each across n_workers processes.

    With share_training_across_epsilons, each job trains once and evaluates every
    epsilon in ensemble_epsilon_list on that trained model (only for trainer oracles
    whose training does not depend on epsilon, i.e. that implement release). Training
    is seeded the same way as when each epsilon gets its own job, so the trained model
    is identical; only the noise draws for epsilons after the first differ.

    build_config must be picklable (a top-level function, or a functools.partial of one)
    and take the ensemble k and epsilon, constructing a fresh ExperimentRunnerConfig,
    including its env, trainer oracle, and MIA classifier.
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


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric per (N, k, epsilon) setting."""
    return (
        table.groupby(by=["N. Trajectories", "Ensemble k", "Ensemble epsilon"], dropna=False)[
            METRIC_COLUMNS
        ]
        .agg(["mean", "std"])
        .round(4)
    )
