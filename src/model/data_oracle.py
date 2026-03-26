from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import random
from collections import defaultdict, deque

import gymnasium as gym
import numpy as np

from config import RandomDataOracleConfig, QLearnerDataOracleConfig

from .generic_model import GenericModel, QLearner

class RandomDataOracle(GenericModel):
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
        super().__init__(config.env)
        
    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        """
        Selects action for given state according to random policy. Note state isn't used.
        
        Args:
            state (Union[int, Tuple]): State to produce an action for.
            
        Returns:
            Union[int, Tuple]: Action for given state.
        """
        return self.env.action_space.sample()
    
class QLearnerDataOracle(QLearner):
    """
    A Data Oracle implementation with a Q Learning policy.
    """
    
    def __init__(self, config: QLearnerDataOracleConfig):
        """
        Initializes a QLearnerDataOracle with the given config info.
        """
        super().__init__(config.env, config.buffer_size, config.buffer_batch_size,
                         config.verbose, config.discount_factor)
        
        self.train_timesteps = 0 # Keep track of training data to allow training multiple times.
        self.learning_starts = config.learning_starts
        
        self.verbose = config.verbose
        
        random.seed(config.random_seed)
    
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
            action = self._select_action(DataOracle._encode_state(state))
            
            next_state, reward, done, _, _ = self.env.step(action)
            
            self.replay_buffer.append((DataOracle._encode_state(state), action, reward, DataOracle._encode_state(next_state)))
            
            if self.train_timesteps >= self.learning_starts:
                batch = random.sample(self.replay_buffer, self.buffer_batch_size)
                
                for traj in batch:
                    self._q_update(*traj)

            if done:
                state, _ = self.env.reset()
                done = False
                
            self.train_timesteps += 1
            self.epsilon *= self.decay_rate
           
        if self.verbose: 
            print("Finished training")