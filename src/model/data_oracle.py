from abc import ABC, abstractmethod
from typing import List, Tuple
import random
from collections import defaultdict, deque

import gymnasium as gym
import numpy as np

from config import RandomDataOracleConfig, QLearnerDataOracleConfig

class DataOracle(ABC):
    """
    An abstract base class for the Data Oracle, which will generate trajectories
    either for training the Trainer Oracle or for external use.
    """
    
    def __init__(self, env: gym.Env, verbose: int=0):
        """
        Initializes a DataOracle with the given environment.
        
        Args:
            env (gym.Env): The environment to generate trajectories for.
            verbose (int): Whether to print status messages when running methods.
        """
        self.env = env
        self.verbose = verbose
        
    @abstractmethod
    def _select_action(self, state):
        pass
        
    def _gen_trajectory(self, T_max: int) -> List[Tuple]:
        """
        Returns a trajectory with no more than T_max transitions.

        Args:
            T_max (int): Maximal number of transitions in a trajectory.

        Returns:
            List[Tuple]: List of (state, action, reward, next_state) tuples.
        """
        
        traj = []

        current_state, _ = self.env.reset()
        for _ in range(T_max):
            # Get the action
            action = self._select_action(current_state)

            # Step through environment
            next_state, reward, done, _, _ = self.env.step(action)

            traj.append((current_state, action, reward, next_state))

            if done:
                break

            current_state = next_state
            
        return traj
    
    def generate_trajectories(
        self, n_trajectories: int, T_max: int, seed: int = None
    ) -> List[List[Tuple]]:
        """
        Returns the requested number of i.i.d. trajectories from the supplied
        environment, stopping early within each trajectory if T_max steps are taken.

        Args:
            n_trajectories (int): Number of trajectories to generate.
            T_max (int): Maximal number of steps in a trajectory.
            seed (int): Optional random seed for reproducibility.

        Returns:
            List[List[Tuple]]
        """
        if self.verbose > 0:
            print(f"Generating {n_trajectories} trajectories.")
        if seed is not None:  # For reuse
            self.env.reset(seed=seed)
            np.random.seed(seed)

        trajectories = []
        for _ in range(n_trajectories):
            traj = self._gen_trajectory(T_max)

            trajectories.append(traj)
            
        if self.verbose > 0:
            print("Finished generating training trajectories.")
            
        return trajectories
        
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
        super().__init__(config.env)
        
    def _select_action(self, state):
        """
        Selects action according to random policy. Note state isn't used.
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
        self.env = config.env
        
        # Q and count table work for anything hashable
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.update_counts = defaultdict(lambda: defaultdict(int))
        
        self.replay_buffer = deque(maxlen=config.buffer_size)
        self.buffer_batch_size = config.buffer_batch_size
        
        self.train_timesteps = 0 # Keep track of training data to allow training multiple times.
        self.learning_starts = config.learning_starts
        self.epsilon = config.epsilon
        self.decay_rate = config.decay_rate
        
        self.discount_factor = config.discount_factor
        
        self.verbose = config.verbose
        
        random.seed(config.random_seed)
    
    
    def _select_action(self, state):
        """
        Selects an action via epsilon-greedy.
        """
        
        if random.random() < self.epsilon:
            # Random choice
            return self.env.action_space.sample()
        else:
            return max(self.q_table[state], key=self.q_table[state].get) if self.q_table[state] else self.env.action_space.sample()
    
    def _q_update(self, state, action, reward, next_state):
        """
        Runs a standard Q learning update.
        """
        n_updates = self.update_counts[state][action]
        old_q = self.q_table[state][action]
        
        alpha = 1 / (1 + n_updates)
        
        new_q = (1 - alpha) * old_q  + alpha * (reward + self.discount_factor * max(self.q_table[next_state].values(), default=0))
        
        self.q_table[state][action] = new_q
        self.update_counts[state][action] = n_updates + 1
        
    def train(self, learn_timesteps: int):
        """
        Trains self for the given number of learning timesteps. Only samples
        from replay buffer once learning_starts is reached.
        """
        print(f"Training Q Learner Data Oracle for {learn_timesteps} timesteps")
        
        state, _ = self.env.reset()
        
        done = False
    
        for _ in range(learn_timesteps):
            action = self._select_action(state)
            
            next_state, reward, done, _, _ = self.env.step(action)
            
            self.replay_buffer.append((state, action, reward, next_state))
            
            if self.train_timesteps >= self.learning_starts:
                batch = random.sample(self.replay_buffer, self.buffer_batch_size)
                
                for traj in batch:
                    self._q_update(*traj)

            if done:
                state, _ = self.env.reset()
                done = False
                
            self.train_timesteps += 1
            self.epsilon *= self.decay_rate
            
        print("Finished training")