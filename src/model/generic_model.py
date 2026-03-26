from typing import List, Tuple, Union
import random
from collections import defaultdict, deque
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

class GenericModel(ABC):
    """
    This is a class storing generic methods for an RL agent to use.
    """
    
    def __init__(self, env: gym.Env, verbose: int=0):
        """
        Initializes a GenericModel with the given environment.
        
        Args:
            env (gym.Env): The environment to generate trajectories for.
            verbose (int): Whether to print status messages when running methods.
        """
        self.env = env
        self.verbose = verbose
    
    @staticmethod
    def _encode_state(state: Union[int, np.ndarray]) -> Union[int, Tuple]:
        """
        Produces a hashable version of the raw state. If it's an int leaves it
        as-is. If it's a numpy array turns it into a tuple of the flattened
        version.
        """
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        else:
            return state
        
    @abstractmethod
    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        """
        Produces an action for the given state. Method for doing that
        left to subclasses.
        
        Args:
            state (Union[int, Tuple]): State to produce an action for.
            
        Returns:
            Union[int, Tuple]: Action for given state.
        """
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
        done = False
        t = 0
        while not done:
            # Get the action
            action = self._select_action(QLearner._encode_state(current_state))

            # Step through environment
            next_state, reward, done, _, _ = self.env.step(action)

            traj.append((QLearner._encode_state(current_state), action, reward, QLearner._encode_state(next_state)))

            current_state = next_state
            
            t += 1
            if t == T_max:
                break
            
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
            
        if self.verbose:
            print("Finished generating training trajectories.")
            
        return trajectories
    

class QLearner(GenericModel):
    """
    This class works to de-duplicate code between any common Q-learning implementations.
    """
    
    def __init__(self, env: gym.Env, buffer_size: int, buffer_batch_size: int, verbose: int, discount_factor: float):
        """
        Initializes a QLearner with the given environment, buffer size, .
        
        Args:
            env (gym.Env): The environment to generate trajectories for.
        """
        super().__init__(env, verbose)
        
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.update_counts = defaultdict(lambda: defaultdict(int))
        
        self.replay_buffer = deque(maxlen=buffer_size)
        self.buffer_batch_size = buffer_batch_size
        
        self.discount_factor = discount_factor
        
    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        """
        Selects an action for the given state via epsilon-greedy.
        
        Args:
            state (Union[int, Tuple]): State to produce an action for.
            
        Returns:
            Union[int, Tuple]: Action for given state.
        """
        
        if random.random() < self.epsilon:
            # Random choice
            return self.env.action_space.sample()
        else:
            return max(self.q_table[state], key=self.q_table[state].get) if self.q_table[state] else self.env.action_space.sample()
    
    def _q_update(self, state: Union[int, Tuple], action: Union[int, Tuple], reward: float, next_state: Union[int, Tuple]) -> None:
        """
        Runs a standard Q learning update with the given (s,a,r,s') info.
        
        Args:
            state (Union[int, Tuple]): Initial state.
            action (Union[int, Tuple]): Transition action.
            reward (float): Transition reward.
            next_state (Union[int, Tuple]): Next state.
        """
        n_updates = self.update_counts[state][action]
        old_q = self.q_table[state][action]
        
        alpha = 1 / (1 + n_updates)
        
        new_q = (1 - alpha) * old_q  + alpha * (reward + self.discount_factor * max(self.q_table[next_state].values(), default=0))
        
        self.q_table[state][action] = new_q
        self.update_counts[state][action] = n_updates + 1
        
    def optimal_state_val(self, state: Union[int, Tuple]) -> float:
        """
        Produces max Q value for given state.
        """
        return max(self.q_table[state].values(), default=0)