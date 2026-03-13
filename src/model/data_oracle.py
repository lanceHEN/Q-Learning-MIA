from abc import ABC, abstractmethod
from typing import List, Tuple

import gymnasium as gym
import numpy as np

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
    
    def __init__(self, env: gym.Env):
        """
        Initializes a RandomDataOracle with the given environment.
        
        Args:
            env (gym.Env): The environment to generate trajectories for.
        """
        super().__init__(env)
        
    def _select_action(self, state):
        """
        Selects action according to random policy. Note state isn't used.
        """
        return self.env.action_space.sample()