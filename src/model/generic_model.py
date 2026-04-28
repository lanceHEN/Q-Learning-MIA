from __future__ import annotations
from typing import List, Tuple, Union, Dict, TYPE_CHECKING
import random
from collections import defaultdict, deque
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    from config import QLearnerConfig

class GenericModel(ABC):
    """
    This is a class storing generic methods for an RL agent to use.
    """
    
    def __init__(self, env: gym.Env, verbose: int=0, state_encoder=None):
        """
        Initializes a GenericModel with the given environment.
        
        Args:
            env (gym.Env): The environment to generate trajectories for.
            verbose (int): Whether to print status messages when running methods.
            state_encoder: Optional function to encode state
        """
        self.env = env
        self.verbose = verbose
        self.state_encoder = state_encoder
    
    def _encode_state(self, state: Union[int, np.ndarray, Dict]) -> Union[int, Tuple]:
        """
        Produces a hashable version of the raw state. If it's an int leaves it
        as-is. If it's a numpy array turns it into a tuple of the flattened
        version.
        """
        if self.state_encoder is not None:
            return self.state_encoder(state)
        
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        elif isinstance(state, dict) and 'image' in state:
            direction = state['direction']
            positions = np.argwhere(state['image'][:,:,0] == 10)
            if len(positions) == 0:
                return (direction, -1, -1)
            
            return (direction,) + tuple(positions[0])
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
            action = self._select_action(self._encode_state(current_state))

            # Step through environment
            next_state, reward, done, _, _ = self.env.step(action)

            traj.append((self._encode_state(current_state), action, reward, self._encode_state(next_state)))

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
    
    def __init__(self, config: QLearnerConfig):
        """
        Initializes a QLearner with the given config information.
        
        Args:
            config (QLearnerConfig): Config info.
        """
        super().__init__(config.env, config.verbose, config.state_encoder)
        
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = config.alpha
        
        self.replay_buffer = deque(maxlen=config.buffer_size)
        self.buffer_batch_size = config.buffer_batch_size
        
        self.discount_factor = config.discount_factor
        
        self.epsilon = config.epsilon
        
    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        """
        Selects an action for the given state (greedy).
        
        Args:
            state (Union[int, Tuple]): State to produce an action for.
            
        Returns:
            Union[int, Tuple]: Action for given state.
        """  
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
        old_q = self.q_table[state][action]
        
        new_q = (1 - self.alpha) * old_q  + self.alpha * (reward + self.discount_factor * max(self.q_table[next_state].values(), default=0))

        
        self.q_table[state][action] = new_q