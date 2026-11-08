from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, TYPE_CHECKING
import pickle
import random
from collections import defaultdict, deque

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import ProgressBarCallback
import torch

if TYPE_CHECKING:
    from config import (DataOracleConfig,
                        QLearnerDataOracleConfig,
                        DQNDataOracleConfig,
                        CustomFixedPolicyDataOracleConfig)

from .generic_model import GenericModel, QLearner

class DataOracle(GenericModel):
    """
    This class provides no additional functionality beyond the GenericModel.
    It exists only for type-checking purposes.
    """

    def _select_action_from_q_vals(self, q_vals, tau):
        """Selects action based on q values via softmax w/ temp tau."""
        if tau <= 0:
            return self.actions[int(np.argmax(q_vals))]
        probs = np.exp((q_vals - np.max(q_vals)) / tau)
        probs /= probs.sum()
        return self.actions[np.random.choice(len(self.actions), p=probs)]
        

class RandomDataOracle(DataOracle):
    """
    A Data Oracle implementation with a random policy.
    """

    def __init__(self, config: DataOracleConfig):
        super().__init__(env=config.env, verbose=config.verbose, state_encoder=config.state_encoder)
        self.config = config

    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        return self.env.action_space.sample()

class CustomFixedPolicyDataOracle(DataOracle):
    """
    A Data Oracle implementation using a fixed policy function which is given on
    construction.
    """

    def __init__(self, config: CustomFixedPolicyDataOracleConfig):
        super().__init__(env=config.env, verbose=config.verbose, state_encoder=config.state_encoder)
        self.config = config

    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        return self.config.action_selector(state)

class QLearnerDataOracle(DataOracle):
    """
    A Data Oracle implementation with a Q Learning policy.
    """

    def __init__(self, config: QLearnerDataOracleConfig):
        super().__init__(env=config.env, verbose=config.verbose, state_encoder=config.state_encoder)
        self.config = config
        self.q_learner = config.q_learner
        random.seed(config.random_seed)

    def reset(self):
        self.q_learner.reset()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({s: dict(qs) for s, qs in self.q_learner.q_table.items()}, f)

    def load(self, path):
        with open(path, "rb") as f:
            q_table = pickle.load(f)
        self.q_learner.q_table = defaultdict(lambda: defaultdict(float),
                                             {s: defaultdict(float, qs) for s, qs in q_table.items()})
        
    def q_vector(self, state: Union[int, Tuple]):
        q = self.q_learner.q_table.get(state, {})
        return np.array([q.get(action, 0.0) for action in self.actions])

    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        q_vals = self.q_vector(state)
        return self._select_action_from_q_vals(q_vals, self.config.tau)

    def _select_action_epsilon_greedy(self, state):
        if random.random() < self.q_learner.epsilon:
            return self.env.action_space.sample()
        return self.q_learner._select_action(state)

    def train(self, learn_timesteps: int):
        """
        Trains self for the given number of learning timesteps. Only samples
        from replay buffer once learning_starts is reached.
        """
        print(f"Training Q Learner Data Oracle for {learn_timesteps} timesteps")

        state, _ = self.env.reset()
        done = False

        for t in range(learn_timesteps):
            action = self._select_action_epsilon_greedy(self._encode_state(state))
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated  # time-limited episodes (e.g. GlucoseEnv) end by truncation
            self.q_learner.replay_buffer.append((self._encode_state(state), action, reward, self._encode_state(next_state)))

            if t >= self.config.learning_starts:
                transition = random.choice(self.q_learner.replay_buffer)
                self.q_learner._q_update(*transition)

            if done:
                state, _ = self.env.reset()
                done = False
            else:
                state = next_state

            self.q_learner.epsilon *= self.config.decay_rate
            self.q_learner.alpha *= self.config.alpha_decay_rate

        if self.verbose:
            print("Finished training")

class DQNDataOracle(DataOracle):
    """
    A DQN-based data oracle for more complex envs.
    """
    def __init__(self, config: DQNDataOracleConfig):
        super().__init__(env=config.env, verbose=config.verbose, state_encoder=config.state_encoder)
        self.config = config
        self.reset()
        
    def save(self, path):
        self.dqn.save(path=path)
        
    def load(self, path):
        self.dqn = DQN.load(path)

    def reset(self):
        kwargs = dict(
            policy=self.config.policy,
            env=self.config.env,
            learning_rate=self.config.learning_rate,
            learning_starts=self.config.learning_starts,
            exploration_fraction=self.config.exploration_fraction,
            exploration_final_eps=self.config.exploration_final_eps,
            batch_size=self.config.batch_size,
            buffer_size=self.config.buffer_size,
            device=self.config.device,
            seed=self.config.seed,
            target_update_interval=self.config.target_update_interval,
            gamma=self.config.gamma
        )
        if self.config.policy_kwargs is not None:
            kwargs['policy_kwargs'] = self.config.policy_kwargs
        self.dqn = DQN(**kwargs)
        
    def q_vector(self, state: Union[int, Tuple]):
        if isinstance(state, tuple):
            state = np.array(state).reshape(self.env.observation_space.shape)
        obs_tensor, _ = self.dqn.policy.obs_to_tensor(state)
        with torch.no_grad():
            q_vals = self.dqn.policy.q_net(obs_tensor).cpu().numpy()[0]
        return q_vals

    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        q_vals = self.q_vector(state)
        return self._select_action_from_q_vals(q_vals, self.config.tau)

    def train(self, learn_timesteps: int):
        """
        Trains self for the given number of learning timesteps.
        """
        self.dqn.learn(total_timesteps=learn_timesteps, callback=ProgressBarCallback())


class DiscretizedPolicyDataOracle(DataOracle):
    """
    Runs a tabular data oracle's policy on an environment with richer observations: each
    observation is mapped to the oracle's state by state_fn to choose the action, while the
    trajectories keep the observations (e.g. a Q-learner trained on GlucoseDiscreteObs
    generating data for learners that see GlucoseEnv's continuous observations).
    """

    def __init__(self, base_oracle: DataOracle, env: gym.Env, state_fn):
        super().__init__(env=env)
        self.base_oracle = base_oracle
        self.state_fn = state_fn

    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        return self.base_oracle._select_action(self.state_fn(state))


class FeatureMappedDataOracle:
    """
    Wraps a data oracle, mapping every state in its trajectories through feature_fn
    (e.g. SepsisStateVectorObs.encode). The behaviour policy still acts on the original
    states, so a tabular data oracle can generate data for a learner that sees features.
    """

    def __init__(self, base_oracle: DataOracle, feature_fn):
        self.base_oracle = base_oracle
        self.feature_fn = feature_fn

    def generate_trajectories(self, n_trajectories: int, T_max: int, seed: int = None) -> List[List[Tuple]]:
        f = self.feature_fn
        return [[(f(s), a, r, f(s2)) for s, a, r, s2 in traj]
                for traj in self.base_oracle.generate_trajectories(n_trajectories, T_max, seed)]
