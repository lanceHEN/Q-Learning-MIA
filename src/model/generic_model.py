from __future__ import annotations
from typing import List, Tuple, Union, Dict, TYPE_CHECKING
import random
from collections import defaultdict, deque
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import itertools
from gymnasium import spaces


if TYPE_CHECKING:
    from config import QLearnerConfig


def all_actions(space: spaces.Space) -> list:
    """Enumerates every action of a finite gym space."""
    if isinstance(space, spaces.Discrete):
        return list(range(int(space.start), int(space.start + space.n)))
    if isinstance(space, spaces.MultiDiscrete):
        return [np.array(a) for a in itertools.product(*[range(int(n)) for n in space.nvec.flatten()])]
    if isinstance(space, spaces.MultiBinary):
        return [np.array(a) for a in itertools.product([0, 1], repeat=int(np.prod(space.shape)))]
    if isinstance(space, spaces.Tuple):
        return list(itertools.product(*[all_actions(s) for s in space.spaces]))
    raise ValueError(f"Cannot enumerate {type(space).__name__}")


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
        self.actions = all_actions(env.action_space)
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
            if t >= T_max:
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
    
class QFunctionLearner(GenericModel, ABC):
    """
    Extends the GenericModel interface with additional abstract methods for
    any class that wants to estimate Q values.
    """
    
    @abstractmethod
    def q_val(self, state: Union[int, Tuple], action: Union[int, Tuple]) -> float:
        """
        Produces Q value for the given state-action pair.
        """
        pass
    
    @abstractmethod
    def reset(self):
        """
        Resets internal state back to its initial values.
        """
        pass
    
class QLearner(QFunctionLearner):
    """
    This class works to de-duplicate code between any common Q-learning implementations.
    We also optionally let it have default values for unseen (s,a), for training or
    inference. This is useful for our ensembling trainer oracle on tabular data.
    """
    
    def __init__(self, config: QLearnerConfig):
        """
        Initializes a QLearner with the given config information.
        
        Args:
            config (QLearnerConfig): Config info.
        """
        super().__init__(env=config.env, verbose=config.verbose, state_encoder=config.state_encoder)
        self.config = config
        self.alpha = config.alpha # changes during training when a data oracle decays it
        self.reset()
        
    def reset(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.replay_buffer = deque(maxlen=self.config.buffer_size)
        self.observed_pairs = set()
        self.epsilon = self.config.epsilon  # changes during training when a data oracle decays it
        
    def q_vals(self, state) -> Dict[Union[int, Tuple], float]:
        # Dense over all actions, so unseen actions get unseen_inference_val even when they
        # have no entry in the table; .get avoids inserting keys on lookup
        qv = self.q_table.get(state, {})
        return {a: qv.get(a, 0.0) if (state, a) in self.observed_pairs else self.config.unseen_inference_val
                for a in self.actions}
    
    def q_val(self, state, action):
        if (state, action) in self.observed_pairs:
            return self.q_table[state][action]
        else:
            return self.config.unseen_inference_val
        
    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        """
        Selects an action for the given state (greedy).
        
        Args:
            state (Union[int, Tuple]): State to produce an action for.
            
        Returns:
            Union[int, Tuple]: Action for given state.
        """  
        return max(self.q_table[state], key=self.q_table[state].get) if self.q_table[state] else self.env.action_space.sample()
    
    def _q_update(self, state: Union[int, Tuple], action: Union[int, Tuple], reward: float, next_state: Union[int, Tuple], ) -> None:
        """
        Runs a standard Q learning update with the given (s,a,r,s') info.
        
        Args:
            state (Union[int, Tuple]): Initial state.
            action (Union[int, Tuple]): Transition action.
            reward (float): Transition reward.
            next_state (Union[int, Tuple]): Next state.
        """
        cfg = self.config
        self.observed_pairs.add((state, action))
        old_q = self.q_table[state][action]
        
        next_qv = self.q_table.get(next_state, {})
        observed = [next_qv.get(a, 0.0) for a in self.actions if (next_state, a) in self.observed_pairs]
        has_unseen = len(observed) < len(self.actions)
        max_next_qv = max(observed + ([cfg.unseen_training_val] if has_unseen else []))
        
        new_q = (1 - self.alpha) * old_q  + self.alpha * (reward + cfg.discount_factor * max_next_qv)
        
        self.q_table[state][action] = new_q