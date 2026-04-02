from collections import defaultdict, deque
import random
from typing import List, Tuple, Union
from abc import ABC, abstractmethod

import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np
import torch

from config import TrainerOracleConfig, QLearnerTrainerOracleConfig, DQNTrainerOracleConfig
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
        super().__init__(config.env, config.verbose)
        
        self.alpha = config.alpha
        self.discount_factor = config.discount_factor
    
    @abstractmethod
    def q_val(self, state: Union[int, Tuple], action: Union[int, Tuple]) -> float:
        """
        Produces Q value for the given state-action pair.
        """
        pass
    
    @abstractmethod
    def train(self, trajectories: List[List[Tuple]], training_steps: int):
        """
        Trains self on the given trajectories for the given number of training_steps.
        
        Starts by adding the trajectories to the replay buffer. For each training step,
        samples a batch of size buffer_batch_size from the buffer and runs Q learning
        on them.
        
        Args:
            trajectories (List[List[Tuple]]): List of trajectories, each being a list
                of (state, action, reward, next_state) tuples.
            learn_timesteps (int): How many steps to train for.
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
        
        Args:
            config (QLearnerTrainerOracleConfig): Config info for QLearnerTrainerOracle. 
        """
        super().__init__(TrainerOracleConfig(
            config.env,
            config.alpha,
            config.discount_factor,
            config.verbose
        ))
        
        self.q_learner = QLearner(config.q_learner_config) # compose w/ q learner
        self.train_timesteps = 0 # Keep track of training data to allow training multiple times.
        
    def q_val(self, state: Union[int, Tuple], action: Union[int, Tuple]) -> float:
        """
        Produces Q value for the given state-action pair.
        """
        return self.q_learner.q_table[state][action]
    
    def train(self, trajectories: List[List[Tuple]], training_steps: int):
        """
        Trains self on the given trajectories for the given number of training_steps.
        
        Starts by adding the trajectories to the replay buffer. For each training step,
        samples a batch of size buffer_batch_size from the buffer and runs Q learning
        on them.
        
        Args:
            trajectories (List[List[Tuple]]): List of trajectories, each being a list
                of (state, action, reward, next_state) tuples.
            learn_timesteps (int): How many steps to train for.
        """
        if self.verbose:
            print(f"Training on {len(trajectories)} trajectories for {training_steps} training steps")
        
        for traj in trajectories:   
            self.q_learner.replay_buffer.extend(traj)
    
        for _ in range(training_steps):
            batch = random.sample(self.q_learner.replay_buffer, self.q_learner.buffer_batch_size)
            for traj in batch:
                self.q_learner._q_update(*traj)
                
            self.train_timesteps += 1
        
        if self.verbose:
            print("Finished training")
            
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
    A DQN-based trainer oracle.
    """
    
    def __init__(self, config: DQNTrainerOracleConfig):
        """
        Initializes with given config info.
        """
        
        super().__init__(TrainerOracleConfig(
            config.env,
            config.alpha,
            config.discount_factor,
            config.verbose
        ))
        
        self.dqn = DQN(policy=config.dqn_config.policy,
                       env=config.dqn_config.env,
                       verbose=config.dqn_config.verbose,
                       learning_rate=config.dqn_config.alpha,
                       learning_starts=config.dqn_config.learning_starts,
                       exploration_fraction=config.dqn_config.exploration_fraction,
                       exploration_final_eps=config.dqn_config.exploration_final_eps,
                       batch_size=config.dqn_config.batch_size,
                       buffer_size=config.dqn_config.buffer_size,
                       optimize_memory_usage=config.dqn_config.optimize_memory_usage)
        
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
    
    def train(self, trajectories: List[List[Tuple]], training_steps: int):
        """
        Trains self on the given trajectories for the given number of training_steps.
        
        Args:
            trajectories (List[List[Tuple]]): List of trajectories, each being a list
                of (state, action, reward, next_state) tuples.
            learn_timesteps (int): How many steps to train for.
        """
        # Test if online works better
        # self.dqn.learn(total_timesteps=training_steps)
        # return
        
        for traj in trajectories:
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
        
    def optimal_state_val(self, state: Union[int, Tuple]) -> float:
        """
        Produces max Q value for given state.
        """
        q_values = self._q_vals(state)
            
        return max(q_values[0])
    
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