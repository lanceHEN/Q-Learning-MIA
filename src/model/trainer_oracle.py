from collections import defaultdict, deque
import random
from typing import List, Tuple, Union

import gymnasium as gym

from config import TrainerOracleConfig
from .q_learner import QLearner

class TrainerOracle(QLearner):
    """
    The Trainer Oracle is the model to conduct MIAs against. It will take
    in training trajectories and run Q learning on those. The Bellman residuals
    may then be used for MIAs.
    """
    
    def __init__(self, config: TrainerOracleConfig):
        """
        Initializes a TrainerOracle with the given config info.
        
        Args:
            config (TrainerOracleConfig): Config info for TrainerOracle. 
        """
        super().__init__(config.env, config.buffer_size, config.buffer_batch_size,
                         config.verbose, config.discount_factor)

        self.train_timesteps = 0 # Keep track of training data to allow training multiple times.
    
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
            self.replay_buffer.extend(traj)
    
        for _ in range(training_steps):
            batch = random.sample(self.replay_buffer, self.buffer_batch_size)
            for traj in batch:
                self._q_update(*traj)
                
            self.train_timesteps += 1
        
        if self.verbose:
            print("Finished training")