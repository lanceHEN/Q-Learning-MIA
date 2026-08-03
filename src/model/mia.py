from typing import List, Tuple, Union
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from .trainer_oracle import TrainerOracle

class MIAClassifier:
    """
    Learns an optimal threshold eta to distinguish training from external trajectory
    Bellman residuals, using the Neyman-Pearson Lemma.
    """

    def __init__(self, trainer_oracle: TrainerOracle):
        self.trainer_oracle = trainer_oracle
        self.eta = None
        self.kde0 = None
        self.kde1 = None

    def _traj_membership_score(self, traj: List[Tuple]) -> float:
        """
        Computes the average squared Bellman residual over the transitions.
        """
        s = 0
        gamma = self.trainer_oracle.discount_factor
        for (state, action, reward, next_state) in traj:
            max_next_val = self.trainer_oracle.optimal_state_val(next_state)
            q_val = self.trainer_oracle.q_val(state, action)
            s += (reward + gamma * max_next_val - q_val) ** 2
        return s / len(traj)

    def _log_likelihood_ratio(self, score):
        p1 = max(float(self.kde1(score)), 1e-300)
        p0 = max(float(self.kde0(score)), 1e-300)
        return np.log(p1) - np.log(p0)

    def _log_likelihood_ratios(self, scores):
        p1 = np.maximum(self.kde1(scores), 1e-300)
        p0 = np.maximum(self.kde0(scores), 1e-300)
        return np.log(p1) - np.log(p0)

    def predict_membership(self, traj: List[Tuple]) -> int:
        """
        Predicts whether trajectory was used in training (1) or not (0) via a
        likelihood ratio test at the learned threshold eta.
        """
        score = self._traj_membership_score(traj)
        ratio = self._log_likelihood_ratio(score)
        return int(ratio > self.eta)

    def predict_memberships(self, trajs: List[List[Tuple]]) -> np.ndarray:
        """
        Predicts membership for each trajectory. Returns a binary array with 1
        for members and 0 for non-members.
        """
        return np.array([self.predict_membership(traj) for traj in trajs])

    def fit(self, train_trajectories: List[List[Tuple]], external_trajectories: List[List[Tuple]], fp_rate: float, experiment_name: str) -> None:
        """
        Fits KDE distributions to member/non-member Bellman residual scores, then
        finds a likelihood ratio threshold eta achieving the given false positive rate.
        """
        member_scores = np.array([self._traj_membership_score(traj) for traj in train_trajectories])
        nonmember_scores = np.array([self._traj_membership_score(traj) for traj in external_trajectories])

        print(f"Member mean: {np.mean(member_scores):.6f}")
        print(f"Non-member mean: {np.mean(nonmember_scores):.6f}")
        print(f"Member max: {np.max(member_scores):.6f}")
        print(f"Non-member max: {np.max(nonmember_scores):.6f}")

        all_scores = np.concatenate((nonmember_scores, member_scores))

        self.kde0 = gaussian_kde(nonmember_scores)
        self.kde1 = gaussian_kde(member_scores)

        # Member vs non-member score distributions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        bins = np.linspace(
            min(np.min(member_scores + 1e-30), np.min(nonmember_scores + 1e-30)),
            max(np.max(member_scores + 1e-30), np.max(nonmember_scores + 1e-30)),
            30
        )
        ax1.hist(member_scores + 1e-30, bins=bins, color='blue', alpha=0.7)
        ax1.set_title('Member')
        ax1.set_xlabel('Membership Score')
        ax2.hist(nonmember_scores + 1e-30, bins=bins, color='orange', alpha=0.7)
        ax2.set_title('Non-member')
        ax2.set_xlabel('Membership Score')
        plt.tight_layout()
        plt.savefig(f"data/plots/member_nonmember_scores/{experiment_name}_score_hist.png")

        # Use ROC curve to find eta for the target fp_rate
        labels = np.concatenate((np.zeros(len(external_trajectories)), np.ones(len(train_trajectories))))
        log_lr_scores = self._log_likelihood_ratios(all_scores)

        fp_rates, _, thresholds = roc_curve(labels, log_lr_scores)
        threshold_idx = np.argmin(abs(fp_rates - fp_rate))
        self.eta = thresholds[threshold_idx]

        # ROC curve (scored by raw Bellman residual, lower = more likely member)
        fpr, tpr, _ = roc_curve(labels, -np.array(all_scores))
        self.roc_auc = auc(fpr, tpr)
        roc_auc = self.roc_auc

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], 'k--', label='random')
        plt.xlabel("FP Rate")
        plt.ylabel("TP Rate (Recall)")
        plt.legend()
        plt.savefig(f"data/plots/roc_curves/{experiment_name}_roc_curve.png")
        plt.close('all')


class SARSAMIA(MIAClassifier):

    def __init__(self, trainer_oracle: TrainerOracle, alpha: float = 0.001, discount_factor: float = 0.999):
        super().__init__(trainer_oracle)
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha
        self.discount_factor = discount_factor

    def _traj_membership_score(self, traj: List[Tuple]) -> float:
        """
        Computes the average squared Bellman residual using the internal SARSA Q-table.
        """
        s = 0
        gamma = self.trainer_oracle.discount_factor
        for (state, action, reward, next_state) in traj:
            max_next_val = max(self.q_table[next_state].values(), default=0)
            q_val = self.q_table[state][action]
            s += (reward + gamma * max_next_val - q_val) ** 2
        return s / len(traj)

    def _sarsa_update(self, state: Union[int, Tuple], action: Union[int, Tuple],
                      reward: float, next_state: Union[int, Tuple], next_action: Union[int, Tuple]) -> None:
        old_q = self.q_table[state][action]
        new_q = (1 - self.alpha) * old_q + self.alpha * (reward + self.discount_factor * self.q_table[next_state][next_action])
        self.q_table[state][action] = new_q

    def train(self, n_trajectories: int, n_epochs: int, T_max: int, seed: int = None):
        """
        Runs SARSA on trajectories from the trainer oracle for the given number of epochs.
        """
        trainer_oracle_trajs = self.trainer_oracle.generate_trajectories(
            n_trajectories=n_trajectories, T_max=T_max, seed=seed
        )

        for epoch in range(n_epochs):
            for traj in trainer_oracle_trajs:
                for i, (state, action, reward, next_state) in enumerate(traj):
                    next_action = traj[i + 1][1] if i < len(traj) - 1 else None
                    self._sarsa_update(state, action, reward, next_state, next_action)
