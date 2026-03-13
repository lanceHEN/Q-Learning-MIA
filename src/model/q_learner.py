from collections import defaultdict, deque
import random
import sys
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
from config import (
    QLearnerConfig
)

class QLearner:
    
    def __init__(self, config: QLearnerConfig):
        """
        Initializes a QLearner with the given config info.
        
        Args:
            config (QLearnerConfig): Config info for QLearner. 
        """
        self._env = config.env
        
        # Q and count table work for anything hashable
        self._q_table = defaultdict(lambda: defaultdict(float))
        self._update_counts = defaultdict(lambda: defaultdict(int))
        
        self._config = config
        
        self._replay_buffer = deque(maxlen=self._config.buffer_size)
        self._buffer_batch_size = self._config.buffer_batch_size
        
        self._train_timesteps = 0 # Keep track of training data to allow training multiple times.
        
        self._discount_factor = self._config.discount_factor
        
        random.seed(self._config.random_seed)
        
    @property
    def env(self):
        return self._env
        
    @property
    def q_table(self):
        return self._q_table
    
    @property
    def update_counts(self):
        return self._update_counts
    
    @property
    def replay_buffer(self):
        return self._replay_buffer
    
    @property
    def buffer_batch_size(self):
        return self._buffer_batch_size
    
    @property
    def train_timesteps(self):
        return self._train_timesteps
    
    @property
    def discount_factor(self):
        return self._discount_factor
    
    def train(self, trajectories: List[Tuple], training_steps: int):
        """
        Trains self on the given trajectories for the given number of training_steps.
        
        Starts by adding the trajectories to the replay buffer. For each training step,
        samples a batch of size buffer_batch_size from the buffer and runs Q learning
        on them.
        
        Args:
            trajectories (List[Tuple]): List of (state, action, reward, next_state)
                tuples.
            learn_timesteps (int): How many steps to train for.
        """
        
        self.replay_buffer.extend(trajectories)
    
        for _ in range(training_steps):
            batch = random.sample(self.replay_buffer, self.buffer_batch_size)
            
                
            self._train_timesteps += 1
            self.epsilon *= self.decay_rate