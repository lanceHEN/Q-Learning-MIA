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
        """
        Initializes an MIAClassifier with the given TrainerOracle.
        
        The TrainerOracle will be used to obtain Bellman residuals.
        
        Args:
            trainer_oracle (TrainerOracle): TrainerOracle to use for Bellman
                residual calculations.
        """
        self.trainer_oracle = trainer_oracle
        self.eta = None
        self.a0 = None
        self.b0 = None
        self.a1 = None
        self.b1 = None
    
    def _traj_membership_score(self, traj: List[Tuple]) -> float:
        """
        Computes the trajectory membership score for a particular trajectory.
        The membership score is the average Bellman residual over the transitions.
        
        Args:
            traj: List[tuple]: List of (state, action, reward, next_state) tuples.
            
        Returns:
            float: Trajectory membership score.
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
        1 for training and 0 for external. This is effectively a likelihood
        ratio hypothesis test with H_1 being member and H_0 being nonmember,
        given by p1(s)/p2(s) > eta, where s is the trajectory membership score
        and eta is the learned threshold for a given FP rate.
        
        Args:
            traj: List[tuple]: List of (state, action, reward, next_state) tuples.
            
        Returns:
            int: 1 if predicted training, else 0.
        """
        score = self._traj_membership_score(traj)
        ratio = stats.gamma.pdf(score, a=self.a1, scale=self.b1) / stats.gamma.pdf(score, a=self.a0, scale=self.b0) 
        
        return ratio > self.eta

    def predict_memberships(self, trajs: List[List[Tuple]]) -> np.ndarray:
        """
        Predicts whether each trajectory was used in training the model, returning
        binary numpy array with 1s for training and 0s for external.
        
        Args:
            trajs (List[List[Tuple]]): List of trajectories, each being a list
                of (state, action, reward, next_state) tuples.
            
        Returns:
            np.ndarray: 1d numpy array of 1s and 0s corresponding to predictions
                for each trajectory. 1 being member, 0 being nonmember.
        """
        return np.array([self.predict_membership(traj) for traj in trajs])
    
    def fit(self, train_trajectories: List[List[Tuple]], external_trajectories: List[List[Tuple]], fp_rate: float) -> None:
        """
        Given the training and external trajectories, learns their Gamma distributions
        and a threshold eta resulting in a likelihood ratio test with the given false positive rate.
        
        Args:
            train_trajectories: List[List[Tuple]]:  List of training trajectories, each being a list
                of (state, action, reward, next_state) tuples.
            external_trajectories: List[List[Tuple]]:  List of external trajectories, each being a list
                of (state, action, reward, next_state) tuples.
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
        plt.axvline(x=self.eta, color="black", label="eta")
        plt.legend()
        plt.show()