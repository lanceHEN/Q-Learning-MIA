import os
from typing import List, Tuple
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

from .trainer_oracle import TrainerOracle

class MIAClassifier:
    """
    Base class for any MIA classifier to inherit with reusable methods. All attacks
    rely on the same mean squared Bellman residual scores.
    """
    
    def __init__(self, trainer_oracle: TrainerOracle):
        """
        Initializes with the given trainer oracle.
        """
        self.trainer_oracle = trainer_oracle
    
    def _traj_membership_score(self, traj: List[Tuple]) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Computes the average squared Bellman residual over the transitions, along
        with the array of raw residuals, the array of raw q_values for each action
        taken, and the array of targets.
        """
        residuals, q_vals, targets = self._residuals(traj)        
        return np.mean((residuals**2)), residuals, q_vals, targets
    
    def _residuals(self, traj: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the array of Bellman residuals over the transitions, along with
        raw q values and targets.
        """
        residuals = []
        q_vals = []
        targets = []
        gamma = self.trainer_oracle.discount_factor
        for (state, action, reward, next_state) in traj:
            max_next_val = self.trainer_oracle.optimal_state_val(next_state)
            q_val = self.trainer_oracle.q_val(state, action)
            target = reward + gamma * max_next_val
            residuals.append(target - q_val)
            targets.append(target)
            q_vals.append(q_val)
            
        return np.array(residuals), np.array(q_vals), np.array(targets)
    
    @abstractmethod
    def predict_membership(self, traj: List[Tuple]) -> int:
        """
        Predicts whether trajectory was used in training (1) or not (0). Exact
        method left to subclasses.
        """
        pass
    
    def predict_memberships(self, trajs: List[List[Tuple]]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Predicts membership for each trajectory. Returns a tuple containing
        (1) a binary array with 1 for members and 0 for non-members,
        (2) an array of scores for each trajectory,
        (3) a list of residuals (each a numpy array of varying length) for each trajectory.
        (4) a list of q values (each a numpy array of varying length) for each trajectory.
        (4) a list of targets (each a numpy array of varying length) for each trajectory.
        """
        pass
    
    @staticmethod
    def plot_residuals(member_residuals, nonmember_residuals, experiment_name):
        """Member vs non-member residual distributions (stacked, shared x-axis)"""
        x_min = min(member_residuals.min(), nonmember_residuals.min())
        x_max = max(member_residuals.max(), nonmember_residuals.max())
        bins = np.linspace(x_min, x_max, 40)
        
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        
        ax_top.hist(member_residuals, bins=bins, color='steelblue', alpha=0.7, density=True)
        ax_top.set_ylabel('Density')
        ax_top.set_title('Member')
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        
        ax_bot.hist(nonmember_residuals, bins=bins, color='coral', alpha=0.7, density=True)
        ax_bot.set_xlabel('Bellman Residual')
        ax_bot.set_ylabel('Density')
        ax_bot.set_title('Non-member')
        ax_bot.spines['top'].set_visible(False)
        ax_bot.spines['right'].set_visible(False)
        
        fig.suptitle('Bellman Residual Distributions', y=1.01)
        fig.tight_layout()
        fig.savefig(f"data/plots/member_nonmember_residuals/{experiment_name}_residual_hist.png",
                            dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    @staticmethod
    def plot_targets(member_targets, nonmember_targets, experiment_name):
        """Member vs non-member target distributions (stacked, shared x-axis)"""
        x_min = min(member_targets.min(), nonmember_targets.min())
        x_max = max(member_targets.max(), nonmember_targets.max())
        bins = np.linspace(x_min, x_max, 40)
        
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        
        ax_top.hist(member_targets, bins=bins, color='steelblue', alpha=0.7, density=True)
        ax_top.set_ylabel('Density')
        ax_top.set_title(f'Member (mean={np.mean(member_targets):.3f}, std={np.std(member_targets):.3f})')
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        
        ax_bot.hist(nonmember_targets, bins=bins, color='coral', alpha=0.7, density=True)
        ax_bot.set_xlabel('Bellman Target')
        ax_bot.set_ylabel('Density')
        ax_bot.set_title(f'Non-member (mean={np.mean(nonmember_targets):.3f}, std={np.std(nonmember_targets):.3f})')
        ax_bot.spines['top'].set_visible(False)
        ax_bot.spines['right'].set_visible(False)
        
        all_targets = np.concatenate([member_targets, nonmember_targets])
        
        fig.suptitle(f'Bellman Target Distributions (mean={np.mean(all_targets):.3f}, std={np.std(all_targets):.3f})', y=1.01)
        fig.tight_layout()
        os.makedirs("data/plots/member_nonmember_targets", exist_ok=True)
        fig.savefig(f"data/plots/member_nonmember_targets/{experiment_name}_target_hist.png",
                            dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    @staticmethod
    def draw_score_strip(ax, scores, color, title, rng):
        """One group's scores as a jittered strip on ax, with a dashed line at their mean."""
        ax.scatter(
            scores,
            rng.uniform(-0.4, 0.4, len(scores)),
            color=color, alpha=0.8, s=16, linewidths=0,
        )
        ax.axvline(scores.mean(), color=color, linestyle='--', linewidth=1.2)
        ax.set_title(title)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    @staticmethod
    def plot_scores(member_scores, nonmember_scores, experiment_name, title='Membership Score Distributions', xlim=None):
        """Member vs non-member scores as parallel strip plots sharing the x-axis (limits xlim if given)."""
        rng = np.random.default_rng(0)
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
        MIAClassifier.draw_score_strip(ax_top, member_scores, 'steelblue', 'Member', rng)
        MIAClassifier.draw_score_strip(ax_bot, nonmember_scores, 'coral', 'Non-member', rng)
        ax_bot.set_xlabel('Membership Score (Mean Squared Bellman Residual)')
        if xlim is not None:
            ax_bot.set_xlim(*xlim)

        fig.suptitle(title, y=1.01)
        fig.tight_layout()
        fig.savefig(f"data/plots/member_nonmember_scores/{experiment_name}_score_hist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    @staticmethod
    def roc_auc(labels, all_scores, experiment_name, save_plot: bool = True):
        """ROC curve (scored by raw score, lower = more likely member). Returns AUC; saves the curve if save_plot."""
        fpr, tpr, _ = roc_curve(labels, -np.array(all_scores))
        roc_auc_val = auc(fpr, tpr)
        if not save_plot:
            return roc_auc_val

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.3f}")
        plt.plot([0, 1], [0, 1], 'k--', label='random')
        plt.xlabel("FP Rate")
        plt.ylabel("TP Rate (Recall)")
        plt.legend()
        plt.savefig(f"data/plots/roc_curves/{experiment_name}_roc_curve.png")
        plt.close('all')
        return roc_auc_val
    
class ClusteringMIAClassifier(MIAClassifier):
    """
    An MIA score-based classifier that relies simply on clustering scores to predict
    membership.
    """
    
    def __init__(self, trainer_oracle: TrainerOracle):
        """
        Initializes with the given trainer oracle. Sets clusters to None until
        predict_memberships is called.
        """
        super().__init__(trainer_oracle)
        self.clusters = None
        
    def predict_membership(self, traj: List[Tuple]) -> int:
        """
        Predicts whether trajectory was used in training (1) or not (0). Note that
        here we must have already had a series of trajectories to cluster on.
        This will throw an error if that has not been done.
        """
        if self.clusters is None:
            raise RuntimeError("predict_memberships must be called first.")

        score, *_ = self._traj_membership_score(traj)

        return self.clusters.predict([[score]]).item()
    
    def predict_memberships(self, trajs: List[List[Tuple]]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        scores = []
        residuals = []
        q_vals = []
        targets = []
        for traj in trajs:
            score, res, qv, tg = self._traj_membership_score(traj)
            scores.append(score)
            residuals.append(res)
            q_vals.append(qv)
            targets.append(tg)
        
        scores = np.array(scores)
        # Note: residuals and q_vals may have different lengths between each trajectory,
        # so we keep them as lists (converting to 2d numpy is impossible w/ different dims)
        
        self.clusters = KMeans(n_clusters=2, random_state=0)
        self.clusters.fit(scores.reshape(-1,1))
        labels = self.clusters.labels_
        # Ensure members are 1 and nonmembers are 0.
        smallest_idx = np.argmin(scores) # 
        if labels[smallest_idx] == 0:
            labels = 1 - labels
        
        return labels, scores, residuals, q_vals, targets

class LRTMIAClassifier(MIAClassifier):
    """
    Learns an optimal threshold eta to distinguish training from external trajectory
    Bellman residuals, using the likelihood ratio test.
    """

    def __init__(self, trainer_oracle: TrainerOracle):
        super().__init__(trainer_oracle)
        self.eta = None
        self.kde0 = None
        self.kde1 = None

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
        score, _ = self._traj_membership_score(traj)
        ratio = self._log_likelihood_ratio(score)
        return int(ratio > self.eta)
    
    def predict_memberships(self, trajs: List[List[Tuple]]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        labels = []
        scores = []
        residuals = []
        q_vals = []
        targets = []
        for traj in trajs:
            score, res, qv, tg = self._traj_membership_score(traj)
            labels.append(self.predict_membership(traj))
            scores.append(score)
            residuals.append(res)
            targets.append(tg)
            q_vals.append(qv)
            
        scores = np.array(scores)
        labels = np.array(labels)
        # Note: residuals and q_vals may have different lengths between each trajectory,
        # so we keep them as lists (converting to 2d numpy is impossible w/ different dims)

        return labels, scores, residuals, q_vals, targets

    def fit(self, train_trajectories: List[List[Tuple]], external_trajectories: List[List[Tuple]], fp_rate: float, experiment_name: str) -> None:
        """
        Fits KDE distributions to member/non-member Bellman residual scores, then
        finds a likelihood ratio threshold eta achieving the given false positive rate.
        """
        member_scores, member_residuals = [], []
        nonmember_scores, nonmember_residuals = [], []

        for traj in train_trajectories:
            score, residuals, _, _ = self._traj_membership_score(traj)
            member_scores.append(score)
            member_residuals.append(residuals)

        for traj in external_trajectories:
            score, residuals, _, _ = self._traj_membership_score(traj)
            nonmember_scores.append(score)
            nonmember_residuals.append(residuals)

        member_scores = np.array(member_scores)
        nonmember_scores = np.array(nonmember_scores)
        member_residuals = np.concatenate(member_residuals)
        nonmember_residuals = np.concatenate(nonmember_residuals)

        all_scores = np.concatenate((nonmember_scores, member_scores))

        self.kde0 = gaussian_kde(nonmember_scores)
        self.kde1 = gaussian_kde(member_scores)

        # Use ROC curve to find eta for the target fp_rate
        labels = np.concatenate((np.zeros(len(external_trajectories)), np.ones(len(train_trajectories))))
        log_lr_scores = self._log_likelihood_ratios(all_scores)

        fp_rates, _, thresholds = roc_curve(labels, log_lr_scores)
        threshold_idx = np.argmin(abs(fp_rates - fp_rate))
        self.eta = thresholds[threshold_idx]
