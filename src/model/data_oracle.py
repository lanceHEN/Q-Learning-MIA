from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import random
from collections import defaultdict, deque

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

from config import RandomDataOracleConfig, QLearnerDataOracleConfig, DQNDataOracleConfig

from .generic_model import GenericModel, QLearner

class DataOracle(GenericModel):
    """
    This class provides no additional functionality beyond the GenericModel.
    It exists only for type-checking purposes.
    """
    pass

class RandomDataOracle(DataOracle):
    """
    A Data Oracle implementation with a random policy.
    """
    
    def __init__(self, config: RandomDataOracleConfig):
        """
        Initializes a RandomDataOracle with the given config.
        
        Args:
            config (RandomDataOracleConfig): Config info for the
                RandomDataOracle.
        """
        super().__init__(config.env, config.verbose)
        
    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        """
        Selects action for given state according to random policy. Note state isn't used.
        
        Args:
            state (Union[int, Tuple]): State to produce an action for.
            
        Returns:
            Union[int, Tuple]: Action for given state.
        """
        return self.env.action_space.sample()
    
class QLearnerDataOracle(DataOracle):
    """
    A Data Oracle implementation with a Q Learning policy.
    """
    
    def __init__(self, config: QLearnerDataOracleConfig):
        """
        Initializes a QLearnerDataOracle with the given config info.
        """
        super().__init__(config.env, config.verbose)
        
        self.q_learner = QLearner(config.q_learner_config)
        
        self.train_timesteps = 0 # Keep track of training data to allow training multiple times.
        self.learning_starts = config.learning_starts
        
        self.decay_rate = config.decay_rate
        
        random.seed(config.random_seed)
    
    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        """
        Produces an action for the given state. Method for doing that
        left to subclasses.
        
        Args:
            state (Union[int, Tuple]): State to produce an action for.
            
        Returns:
            Union[int, Tuple]: Action for given state.
        """
        return self.q_learner._select_action(state)
        
    def _select_action_epsilon_greedy(self, state):
        """
        Select an action via epsilon-greedy, for given state.
        """
        if random.random() < self.q_learner.epsilon:
            # Random choice
            return self.env.action_space.sample()
        else:
            return self._select_action(state)
    
    def train(self, learn_timesteps: int):
        """
        Trains self for the given number of learning timesteps. Only samples
        from replay buffer once learning_starts is reached.
        
        Args:
            learn_timesteps (int): Number of timesteps to learn.
        """
        print(f"Training Q Learner Data Oracle for {learn_timesteps} timesteps")
        
        state, _ = self.env.reset()
        
        done = False
    
        for _ in range(learn_timesteps):
            action = self._select_action_epsilon_greedy(self._encode_state(state))
            
            next_state, reward, done, _, _ = self.env.step(action)
            
            self.q_learner.replay_buffer.append((self._encode_state(state), action, reward, self._encode_state(next_state)))
            
            if self.train_timesteps >= self.learning_starts:
                batch = random.sample(self.q_learner.replay_buffer, self.q_learner.buffer_batch_size)
                
                for traj in batch:
                    self.q_learner._q_update(*traj)

            if done:
                state, _ = self.env.reset()
                done = False
                
            self.train_timesteps += 1
            self.q_learner.epsilon *= self.decay_rate
           
        if self.verbose: 
            print("Finished training")

# For now the overlap with the DQN trainer oracle isn't enough to abstract out
# their similarities
class DQNDataOracle(DataOracle):
    """
    A DQN-based data oracle for more complex envs.
    """
    
    
    def __init__(self, config: DQNDataOracleConfig):
        """
        Initializes a DQNDataOracle with the given config info.
        """
        super().__init__(config.env, config.verbose)
        
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
        
    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        """
        Selects action for given state according to DQN policy. Note state isn't used.
        
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
    
    def train(self, learn_timesteps: int):
        """
        Trains self for the given number of learning timesteps. Only samples
        from replay buffer once learning_starts is reached.
        
        Args:
            learn_timesteps (int): Number of timesteps to learn.
        """
        self.dqn.learn(total_timesteps=learn_timesteps)
    