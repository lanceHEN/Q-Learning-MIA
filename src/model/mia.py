from typing import List, Tuple

import numpy as np
from scipy import stats

from model import TrainerOracle

class MIAClassifier:
    """
    Learns an optimal threshold eta to distinguish training from external trajectory
    Bellman residuals.
    """
    
    
    def __init__(self, trainer_oracle: TrainerOracle):
        self.trainer_oracle = trainer_oracle
        self.eta = None
    
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
        membership_score = self._traj_membership_score(traj)
        return int(membership_score < self.eta)

    def predict_memberships(self, trajs: List[List[Tuple]]) -> np.ndarray:
        """
        Predicts whether each trajectory was used in training the model, returning
        binary numpy array with 1s for training and 0s for external.
        """
        return np.array([self.predict_membership(traj) for traj in trajs])
    
    def fit(self, train_trajectories: List[List[Tuple]], external_trajectories: List[List[Tuple]], fp_rate: float) -> None:
        """
        Given the training and external trajectories, learns a threshold eta,
        such that if a given membership score is below eta it is considered
        a training trajectory, and is otherwise an external trajectory.
        
        This is accomplished by running the Neyman-Pearson lemma with the given
        false positive rate fp_rate, applied to the distributions according to
        the member and non-member hypotheses.
        
        These distributions are assumed to be gamma.
        """
        # Distributions for train and external membership scores
        nonmember_scores = np.array([self._traj_membership_score(traj) for traj in external_trajectories])
        
        # Get Gamma params for nonmember (H_0) dist
        alpha, _, beta = stats.gamma.fit(nonmember_scores, floc=0)
        
        # Inverse cdf to fp_rate
        self.eta = stats.gamma.ppf(fp_rate, a=alpha, scale=beta)