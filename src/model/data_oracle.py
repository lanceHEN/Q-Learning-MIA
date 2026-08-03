from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, TYPE_CHECKING
import random
from collections import defaultdict, deque

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import ProgressBarCallback

if TYPE_CHECKING:
    from config import (RandomDataOracleConfig,
                        QLearnerDataOracleConfig,
                        DQNDataOracleConfig,
                        CustomFixedPolicyDataOracleConfig)

from .generic_model import GenericModel, QLearner

class DataOracle(GenericModel):
    """
    This class provides no additional functionality beyond the GenericModel.
    It exists only for type-checking purposes.
    """
    pass

class RandomDataOracle(DataOracle):
    """
    A Data Oracle implementation with a random policy.
    """

    def __init__(self, config: RandomDataOracleConfig):
        super().__init__(env=config.env, verbose=config.verbose, state_encoder=config.state_encoder)

    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        return self.env.action_space.sample()

class CustomFixedPolicyDataOracle(DataOracle):
    """
    A Data Oracle implementation using a fixed policy function which is given on
    construction.
    """

    def __init__(self, config: CustomFixedPolicyDataOracleConfig):
        super().__init__(env=config.env, verbose=config.verbose, state_encoder=config.state_encoder)
        self.action_selector = config.action_selector

    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        return self.action_selector(state)

class QLearnerDataOracle(DataOracle):
    """
    A Data Oracle implementation with a Q Learning policy.
    """

    def __init__(self, config: QLearnerDataOracleConfig):
        super().__init__(env=config.env, verbose=config.verbose, state_encoder=config.state_encoder)
        self.q_learner = config.q_learner
        self.learning_starts = config.learning_starts
        self.decay_rate = config.decay_rate
        random.seed(config.random_seed)

    def reset(self):
        self.q_learner.reset()

    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        return self._select_action_epsilon_greedy(state)

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
            next_state, reward, done, _, _ = self.env.step(action)
            self.q_learner.replay_buffer.append((self._encode_state(state), action, reward, self._encode_state(next_state)))

            if t >= self.learning_starts:
                transition = random.choice(self.q_learner.replay_buffer)
                self.q_learner._q_update(*transition)

            if done:
                state, _ = self.env.reset()
                done = False
            else:
                state = next_state

            self.q_learner.epsilon *= self.decay_rate

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

    def reset(self):
        self.dqn = DQN(
            policy=self.config.policy,
            env=self.config.env,
            learning_rate=self.config.learning_rate,
            learning_starts=self.config.learning_starts,
            exploration_fraction=self.config.exploration_fraction,
            exploration_final_eps=self.config.exploration_final_eps,
            batch_size=self.config.batch_size,
            buffer_size=self.config.buffer_size,
            device=self.config.device,
        )

    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        if isinstance(state, tuple):
            state = np.array(state).reshape(self.env.observation_space.shape)
        action, _ = self.dqn.predict(state, deterministic=True)
        if action.ndim == 0:
            action = int(action)
        return action

    def train(self, learn_timesteps: int):
        """
        Trains self for the given number of learning timesteps.
        """
        self.dqn.learn(total_timesteps=learn_timesteps, callback=ProgressBarCallback())
