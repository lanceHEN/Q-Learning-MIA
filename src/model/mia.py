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
        """
        Initializes an MIAClassifier with the given TrainerOracle.
        
        The TrainerOracle will be used to obtain Bellman residuals.
        
        Args:
            trainer_oracle (TrainerOracle): TrainerOracle to use for Bellman
                residual calculations.
            
        """
        self.trainer_oracle = trainer_oracle
        self.eta = None
        self.kde0 = None
        self.kde1 = None
    
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
        gamma = self.trainer_oracle.discount_factor
        for (state, action, reward, next_state) in traj:
            max_next_val = self.trainer_oracle.optimal_state_val(next_state)
            q_val = self.trainer_oracle.q_val(state, action)
            s += (reward + gamma*max_next_val - q_val)**2
            
        return s / len(traj)
    
    def _log_likelihood_ratio(self, score):
        p1 = max(float(self.kde1(score)), 1e-300)  # add floor
        p0 = max(float(self.kde0(score)), 1e-300)
        return np.log(p1) - np.log(p0)
    
    # Vectorized version
    def _log_likelihood_ratios(self, scores):
        p1 = np.maximum(self.kde1(scores), 1e-300)
        p0 = np.maximum(self.kde0(scores), 1e-300)
        return np.log(p1) - np.log(p0)
    
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
        ratio = self._log_likelihood_ratio(score)
        
        return int(ratio > self.eta)

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
    
    def fit(self, train_trajectories: List[List[Tuple]], external_trajectories: List[List[Tuple]], fp_rate: float, experiment_name: str) -> None:
        """
        Given the training and external trajectories, learns their Gamma distributions
        and a threshold eta resulting in a likelihood ratio test with the given false positive rate.
        
        Args:
            train_trajectories: List[List[Tuple]]:  List of training trajectories, each being a list
                of (state, action, reward, next_state) tuples.
            external_trajectories: List[List[Tuple]]:  List of external trajectories, each being a list
                of (state, action, reward, next_state) tuples.
            experiment_name (str): Name of experiment to save figures with.
        """
        # Distributions for train and external membership scores
        member_scores = np.array([self._traj_membership_score(traj) for traj in train_trajectories])
        nonmember_scores = np.array([self._traj_membership_score(traj) for traj in external_trajectories])
        
        print(f"Member mean: {np.mean(member_scores):.6f}")
        print(f"Non-member mean: {np.mean(nonmember_scores):.6f}")
        print(f"Member max: {np.max(member_scores):.6f}")
        print(f"Non-member max: {np.max(nonmember_scores):.6f}")
        
        # For stability
        #member_scores = np.log(member_scores)
        #nonmember_scores = np.log(nonmember_scores)
        
        #print(member_scores[:10])
        #print(nonmember_scores[:10])
        
        all_scores = np.concatenate((nonmember_scores, member_scores))
        
        # Get kde params for nonmember (H_0) dist
        self.kde0 = gaussian_kde(nonmember_scores)
        
        # Same for member H_1
        self.kde1 = gaussian_kde(member_scores)
        
        # Member vs nonmember sores
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
        
        # Use ROC to solve for eta
        labels = np.concatenate((np.zeros(len(external_trajectories)), np.ones(len(train_trajectories))))
        
        # Necessary for numerical stability
        log_lr_scores = self._log_likelihood_ratios(all_scores)
        
        #valid = np.isfinite(lr_scores)
        #lr_scores = lr_scores[valid]
        #labels = labels[valid]
        
        fp_rates, _, thresholds = roc_curve(labels, log_lr_scores)
    
        threshold_idx = np.argmin(abs(fp_rates - fp_rate))
        self.eta = thresholds[threshold_idx]
        
        # Prints learned params
        
        # print(self.a0, self.b0, self.a1, self.b1)
        
        # negate because lower score = more likely member
        fpr, tpr, thresholds = roc_curve(labels, -np.array(all_scores))
        roc_auc = auc(fpr, tpr)

        # ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1], [0,1], 'k--', label='random')
        plt.xlabel("FP Rate")
        plt.ylabel("TP Rate (Recall)")
        plt.legend()
        plt.savefig(f"data/plots/roc_curves/{experiment_name}_roc_curve.png")
        plt.close('all')
        
class SARSAMIA(MIAClassifier):
    
    def __init__(self, trainer_oracle: TrainerOracle, alpha: float = 0.001, discount_factor: float=0.999):
        super().__init__(trainer_oracle)
        
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha
        self.discount_factor = discount_factor
    
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
        gamma = self.trainer_oracle.discount_factor
        for (state, action, reward, next_state) in traj:
            max_next_val = max(self.q_table[next_state].values(), default=0)
            q_val = self.q_table[state][action]
            s += (reward + gamma*max_next_val - q_val)**2
            
        return s / len(traj)
    
    def _sarsa_update(self, state: Union[int, Tuple], action: Union[int, Tuple],
                      reward: float, next_state: Union[int, Tuple], next_action: Union[int, Tuple]) -> None:
        """
        Runs a standard SARSA update with the given (s,a,r,s',a') info.
        
        Args:
            state (Union[int, Tuple]): Initial state.
            action (Union[int, Tuple]): Transition action.
            reward (float): Transition reward.
            next_state (Union[int, Tuple]): Next state.
            next_action (Union[int, Tuple]): Next action.
        """
        old_q = self.q_table[state][action]
        
        new_q = (1 - self.alpha) * old_q  + self.alpha * (reward + self.discount_factor * self.q_table[next_state][next_action])
        
        self.q_table[state][action] = new_q
    
    def train(self, n_trajectories: int, n_epochs: int, T_max: int, seed: int = None):
        """
        Runs SARSA with the given number of trajectories, looping the given number
        of epochs to get from the trainer oracle, T_max and seed.
        
        Args:
            n_trajectories (int): Number of trajectories to get from the trainer oracle.
            n_epochs (int): Number of times to loop through the trajectories.
            T_max (int): Max trajectory length.
            seed (int): Optional random seed.
        """
        
        trainer_oracle_trajs = self.trainer_oracle.generate_trajectories(n_trajectories=n_trajectories, T_max=T_max, seed=seed)
        
        for epoch in range(n_epochs):
            total_change = 0
            n_updates = 0 
            for traj in trainer_oracle_trajs:
                n = len(traj)
        
                for i in range(n):
                    state = traj[i][0]
                    action = traj[i][1]
                    reward = traj[i][2]
                    next_state = traj[i][3]
                    if i < n - 1: 
                        next_action = traj[i+1][1]
                    else: 
                        next_action = None

                    self._sarsa_update(state, action, reward, next_state, next_action)