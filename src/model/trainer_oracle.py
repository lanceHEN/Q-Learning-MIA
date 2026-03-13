from collections import defaultdict, deque
import random
from typing import List, Tuple

import gymnasium as gym

from config import TrainerOracleConfig

class TrainerOracle:
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
        # Q and count table work for anything hashable
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.update_counts = defaultdict(lambda: defaultdict(int))
        
        self.verbose = config.verbose
        
        self.replay_buffer = deque(maxlen=config.buffer_size)
        self.buffer_batch_size = config.buffer_batch_size
        
        self.train_timesteps = 0 # Keep track of training data to allow training multiple times.
        
        self.discount_factor = config.discount_factor

    def _q_update(self, state, action, reward, next_state):
        """
        Runs a standard Q learning update.
        """
        #print(state)
        #print(action)
        n_updates = self.update_counts[state][action]
        old_q = self.q_table[state][action]
        
        alpha = 1 / (1 + n_updates)
        
        new_q = (1 - alpha) * old_q  + alpha * (reward + self.discount_factor * self.optimal_state_val(next_state))
        
        self.q_table[state][action] = new_q
        self.update_counts[state][action] = n_updates + 1
        
    def optimal_state_val(self, state):
        return max(self.q_table[self, state].values(), default=0)
        
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
        if self.verbose == 1:
            print(f"Training on {len(trajectories)} trajectories for {training_steps} training steps")
        
        for traj in trajectories:   
            self.replay_buffer.extend(traj)
    
        for _ in range(training_steps):
            batch = random.sample(self.replay_buffer, self.buffer_batch_size)
            for traj in batch:
                self._q_update(*traj)
                
            self.train_timesteps += 1
            
        print("Finished training")