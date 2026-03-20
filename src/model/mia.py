from typing import List, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

from model import TrainerOracle

class MIAClassifier:
    """
    Learns an optimal threshold eta to distinguish training from external trajectory
    Bellman residuals, using the Neyman-Pearson Lemma.
    """
    
    
    def __init__(self, trainer_oracle: TrainerOracle):
        self.trainer_oracle = trainer_oracle
        self.eta = None
        self.a0 = None
        self.b0 = None
        self.a1 = None
        self.b1 = None
    
    def _traj_membership_score(self, traj: List[Tuple]) -> float:
        """
        Computes the trajectory membership score for a particular trajectory.
        """
        
        s = 0
        for (state, action, reward, next_state) in traj:
            max_next_val = self.trainer_oracle.optimal_state_val(next_state)
            gamma = self.trainer_oracle.discount_factor
            q_val = self.trainer_oracle.q_table[state][action]
        
            s += (reward + gamma*max_next_val - q_val)**2
            
        return s / len(traj)
    
    def predict_membership(self, traj: List[Tuple]) -> int:
        """
        Predicts whether trajectory was used in training the model, returning
        1 for training and 0 for external.
        """
        score = self._traj_membership_score(traj)
        ratio = stats.gamma.pdf(score, a=self.a1, scale=self.b1) / stats.gamma.pdf(score, a=self.a0, scale=self.b0) 
        
        return ratio > self.eta

    def predict_memberships(self, trajs: List[List[Tuple]]) -> np.ndarray:
        """
        Predicts whether each trajectory was used in training the model, returning
        binary numpy array with 1s for training and 0s for external.
        """
        return np.array([self.predict_membership(traj) for traj in trajs])
    
    def fit(self, train_trajectories: List[List[Tuple]], external_trajectories: List[List[Tuple]], fp_rate: float) -> None:
        """
        Given the training and external trajectories, learns their Gamma distributions
        and a threshold eta resulting in a likelihood ratio test with the given false positive rate.
        """
        # Distributions for train and external membership scores
        member_scores = np.array([self._traj_membership_score(traj) for traj in train_trajectories])
        nonmember_scores = np.array([self._traj_membership_score(traj) for traj in external_trajectories])
        all_scores = np.concatenate((nonmember_scores, member_scores))
        
        # Get Gamma params for nonmember (H_0) dist
        self.a0, _, self.b0 = stats.gamma.fit(nonmember_scores, floc=0)
        
        # Same for member H_1
        self.a1, _, self.b1 = stats.gamma.fit(member_scores, floc=0)
        
        # Use ROC to solve for eta
        labels = np.concatenate((np.zeros(len(external_trajectories)), np.ones(len(train_trajectories))))
        
        lr_scores = stats.gamma.pdf(all_scores, a=self.a1, scale=self.b1) / stats.gamma.pdf(all_scores, a=self.a0, scale=self.b0)
        
        fp_rates, _, thresholds = roc_curve(labels, lr_scores)
    
        threshold_idx = np.argmin(abs(fp_rates - fp_rate))
        self.eta = thresholds[threshold_idx]
        
        xp = np.linspace(0,np.max(member_scores))
        yp0 = stats.gamma.pdf(xp, a=self.a0, scale=self.b0)
        yp1 = stats.gamma.pdf(xp, a=self.a1, scale=self.b1)
        
        print(self.a0, self.b0, self.a1, self.b1)
        
        plt.plot(xp, yp0, color="red", label="p0")
        plt.plot(xp, yp1, color="blue", label="p1")
        plt.legend()
        plt.show()