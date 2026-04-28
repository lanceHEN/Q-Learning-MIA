from __future__ import annotations
from collections import defaultdict, deque
import random
from typing import List, Tuple, Union, TYPE_CHECKING
from abc import ABC, abstractmethod

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch

if TYPE_CHECKING:
    from config import (
        TrainerOracleConfig,
        QLearnerTrainerOracleConfig,
        DeepTrainerOracleConfig,
        DeepOfflineTrainerOracleConfig,
        DeepOnlineTrainerOracleConfig
    )
from .generic_model import GenericModel, QLearner

class TrainerOracle(GenericModel):
    """
    Abstract class for Trainer Oracle implementations, including
    scaffolding for training and Bellman residual calculations.
    """
    
    def __init__(self, config: TrainerOracleConfig):
        """
        Initializes a TrainerOracle with the given config.
        """
        super().__init__(config.env, config.verbose, config.state_encoder)
        
        self.alpha = config.alpha
        self.discount_factor = config.discount_factor
    
    @abstractmethod
    def q_val(self, state: Union[int, Tuple], action: Union[int, Tuple]) -> float:
        """
        Produces Q value for the given state-action pair.
        """
        pass
    
    @abstractmethod
    def train(self, training_steps: int, n_train: int, n_external: int, T_max: int, seed: int = None) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Trains self for the given number of training_steps, returning the given
        number of training and external trajectories.
        
        Args:
            learn_timesteps (int): How many steps to train for.
            n_train (int): Number of training trajectories to generate.
            n_external (int): Number of external trajectories to generate.
            T_max (int): Maximal number of steps in a trajectory.
            seed (int): Optional random seed for reproducibility.
        Returns:
            Tuple[List[Tuple], List[Tuple]]: Training and external trajectories,
                respectively.
        """
        pass
    
    @abstractmethod
    def optimal_state_val(state: Union[int, Tuple]):
        """
        Produces optimal Q value for a given state.
        """
        pass

class QLearnerTrainerOracle(TrainerOracle):
    """
    The QLearnerTrainerOracle is the model to conduct MIAs against. It will take
    in training trajectories and run Q learning on those. The Bellman residuals
    may then be used for MIAs.
    """
    
    def __init__(self, config: QLearnerTrainerOracleConfig):
        """
        Initializes a QLearnerTrainerOracle with the given config info.
        
        NOTE: The Data Oracle should be trained before being fed in.
        
        Args:
            config (QLearnerTrainerOracleConfig): Config info for QLearnerTrainerOracle. 
        """
        from config import TrainerOracleConfig
        
        super().__init__(TrainerOracleConfig(
            config.env,
            config.alpha,
            config.discount_factor,
            config.verbose,
            config.state_encoder
        ))
        
        self.data_oracle = config.data_oracle
        self.q_learner = config.q_learner # compose w/ q learner
        
    def q_val(self, state: Union[int, Tuple], action: Union[int, Tuple]) -> float:
        """
        Produces Q value for the given state-action pair.
        """
        return self.q_learner.q_table[state][action]
    
    def train(self, training_steps: int, n_train: int, n_external: int, T_max: int, seed: int = None) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Trains self for the given number of training_steps, returning the given
        number of training and external trajectories.
        
        Args:
            learn_timesteps (int): How many steps to train for.
            n_train (int): Number of training trajectories to generate.
            n_external (int): Number of external trajectories to generate.
            T_max (int): Maximal number of steps in a trajectory.
            seed (int): Optional random seed for reproducibility.
        Returns:
            Tuple[List[Tuple], List[Tuple]]: Training and external trajectories,
                respectively.
        """
        
        trajectories = self.data_oracle.generate_trajectories(n_train + n_external, T_max, seed)
        
        train_trajectories = trajectories[:n_train]
        external_trajectories = trajectories[n_train:]
        
        if self.verbose:
            print(f"Training on {len(train_trajectories)} trajectories for {training_steps} training steps")
        
        for traj in train_trajectories:
            self.q_learner.replay_buffer.extend(traj)
            
        buffer_list = list(self.q_learner.replay_buffer)
    
        for train_timesteps in range(training_steps):
            transition = random.choice(buffer_list)
            
            self.q_learner._q_update(*transition)
        
        if self.verbose:
            print("Finished training")
            
        return train_trajectories, external_trajectories
            
    def optimal_state_val(self, state: Union[int, Tuple]) -> float:
        """
        Produces max Q value for given state.
        """
        return max(self.q_learner.q_table[state].values(), default=0)
    
    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        """
        Selects an action for the given state.
        
        Args:
            state (Union[int, Tuple]): State to produce an action for.
            
        Returns:
            Union[int, Tuple]: Action for given state.
        """
        return self.q_learner._select_action(state)
            
class DeepTrainerOracle(TrainerOracle):
    """
    Abstracts common implementation details between the Offline/Online
    """
    
    def __init__(self, config: DeepTrainerOracleConfig):
        """
        Initializes with given config info.
        """
        from config import TrainerOracleConfig
        
        super().__init__(TrainerOracleConfig(
            config.env,
            config.alpha,
            config.discount_factor,
            config.verbose,
            config.state_encoder
        ))
        
        self.dqn = config.dqn
        
    def _q_vals(self, state: Union[int, Tuple]):
        """
        Produces [1, n_actions] array of Q values for given state.
        """
        if isinstance(state, tuple):
            state = np.array(state)
        
        obs_tensor, _ = self.dqn.policy.obs_to_tensor(state)
        
        with torch.no_grad():
            q_values = self.dqn.policy.q_net(obs_tensor)  # shape: [1, n_actions]
            
        return q_values
    
    def q_val(self, state: Union[int, Tuple], action: Union[int, Tuple]) -> float:
        """
        Produces Q value for the given state-action pair.
        """
        q_values = self._q_vals(state)
        
        return q_values[0, action].item()
        
    def optimal_state_val(self, state: Union[int, Tuple]) -> float:
        """
        Produces max Q value for given state.
        """
        q_values = self._q_vals(state)
            
        return q_values[0].max().item()
    
    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        """
        Selects an action for the given state.
        
        Args:
            state (Union[int, Tuple]): State to produce an action for.
            
        Returns:
            Union[int, Tuple]: Action for given state.
        """
        if isinstance(state, tuple):
            state = np.array(state)
        action, _ = self.dqn.predict(state, deterministic=True)
        
        if action.ndim == 0:
            action = int(action)
        return action

class DeepOfflineTrainerOracle(DeepTrainerOracle):
    """
    A DQN-based trainer oracle with offline learning.
    """
    
    def __init__(self, config: DeepOfflineTrainerOracleConfig):
        """
        Initializes with given config info.
        """
        
        from config import DeepTrainerOracleConfig
        
        super().__init__(DeepTrainerOracleConfig(
            config.env,
            config.alpha,
            config.discount_factor,
            config.verbose,
            config.dqn,
            config.state_encoder
        ))
        
        self.data_oracle = config.data_oracle
        
    def train(self, training_steps: int, n_train: int, n_external: int, T_max: int, seed: int = None) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Trains self for the given number of training_steps, returning the given
        number of training and external trajectories.
        
        Args:
            learn_timesteps (int): How many steps to train for.
            n_train (int): Number of training trajectories to generate.
            n_external (int): Number of external trajectories to generate.
            T_max (int): Maximal number of steps in a trajectory.
            seed (int): Optional random seed for reproducibility.
        Returns:
            Tuple[List[Tuple], List[Tuple]]: Training and external trajectories,
                respectively.
        """
        trajectories = self.data_oracle.generate_trajectories(n_train + n_external, T_max, seed)
        
        train_trajectories = trajectories[:n_train]
        external_trajectories = trajectories[n_train:]
        
        if self.verbose:
            print(f"Training on {len(train_trajectories)} trajectories for {training_steps} training steps")
        
        for traj in train_trajectories:
            n = len(traj)
            
            for i in range(n):
                if i == n - 1:
                    done = True
                else:
                    done = False
                
                state = traj[i][0]
                action = traj[i][1]
                reward = traj[i][2]
                next_state = traj[i][3]
                
                self.dqn.replay_buffer.add(
                    np.array(state),
                    np.array(next_state),
                    np.array([action]),
                    np.array([reward]),
                    np.array([done]),
                    [{}]
                )
        
        self.dqn.learn(training_steps)
        
        if self.verbose:
            print("Finished training")
            
        return train_trajectories, external_trajectories
    
class DeepOnlineTrainerOracle(DeepTrainerOracle):
    """
    A DQN-based trainer oracle with online learning.
    """
    
    def __init__(self, config: DeepOnlineTrainerOracleConfig):
        """
        Initializes with given config info.
        """
        
        from config import DeepTrainerOracleConfig
        
        super().__init__(DeepTrainerOracleConfig(
            config.env,
            config.alpha,
            config.discount_factor,
            config.verbose,
            config.dqn,
            config.state_encoder
        ))

    def train(self, training_steps: int, n_train: int, n_external: int, T_max: int, seed: int = None) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Trains self for the given number of training_steps, returning the given
        number of training and external trajectories.
        
        Args:
            learn_timesteps (int): How many steps to train for.
            n_train (int): Number of training trajectories to generate.
            n_external (int): Number of external trajectories to generate.
            T_max (int): Maximal number of steps in a trajectory.
            seed (int): Optional random seed for reproducibility.
        Returns:
            Tuple[List[Tuple], List[Tuple]]: Training and external trajectories,
                respectively.
        """
        
        train_trajectories = []
        current_traj = []

        # Custom callback to record transitions
        class TrajectoryRecorder(BaseCallback):
            def _on_step(self):
                def format_value(val):
                    if val.ndim == 1:
                        return val[0]
                    else:
                        return val
            
                current_traj.append((
                    format_value(self.locals['self']._last_obs),
                    format_value(self.locals['actions']),
                    format_value(self.locals['rewards']),
                    format_value(self.locals['new_obs'])
                ))
                if self.locals['dones']:
                    train_trajectories.append(current_traj.copy())
                    current_traj.clear()
                return True
   
        self.dqn.learn(total_timesteps=training_steps, callback=TrajectoryRecorder())
        
        # Gen external trajectories from policy
        external_trajectories = self.generate_trajectories(n_external, T_max, seed)
        
        return train_trajectories[-n_train:], external_trajectories