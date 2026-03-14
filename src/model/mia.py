from typing import List, Tuple

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
    
    def predict_membership(self, traj: List[Tuple]):
        """
        Predicts whether trajectory was used in training the model, returning
        1 for training and 0 for external.
        """
        
        membership_score = self._traj_membership_score(traj)
        return int(membership_score < self.eta)
    
    def train(self, train_trajectories, external_trajectories):
        
        

