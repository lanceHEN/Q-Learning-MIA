from __future__ import annotations
import random
from typing import List, Tuple, Union, TYPE_CHECKING
from abc import ABC, abstractmethod

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
import numpy as np
import torch

if TYPE_CHECKING:
    from config import (
        TrainerOracleConfig,
        QLearnerTrainerOracleConfig,
        DeepTrainerOracleConfig,
        DeepOfflineTrainerOracleConfig,
        DeepOnlineTrainerOracleConfig
    )
from .generic_model import GenericModel, QLearner


class TrajectoryRecorder(BaseCallback):
    """
    SB3 callback that records transitions into episode-level trajectories
    during online DQN training.
    """

    def __init__(self):
        super().__init__()
        self.train_trajectories: List[List[Tuple]] = []
        self._current_traj: List[Tuple] = []

    def _on_step(self) -> bool:
        def _squeeze(val):
            return val[0] if hasattr(val, "ndim") and val.ndim == 1 else val

        self._current_traj.append((
            _squeeze(self.model._last_obs),
            _squeeze(self.locals["actions"]),
            _squeeze(self.locals["rewards"]),
            _squeeze(self.locals["new_obs"]),
        ))
        if self.locals["dones"][0]:
            self.train_trajectories.append(self._current_traj.copy())
            self._current_traj.clear()
        return True


class TrainerOracle(GenericModel):
    """
    Abstract class for Trainer Oracle implementations, including
    scaffolding for training and Bellman residual calculations.
    """

    def __init__(self, config: TrainerOracleConfig):
        super().__init__(config.env, config.verbose, config.state_encoder)
        self.alpha = config.alpha
        self.discount_factor = config.discount_factor

    @abstractmethod
    def q_val(self, state: Union[int, Tuple], action: Union[int, Tuple]) -> float:
        """
        Produces Q value for the given state-action pair.
        """
        pass

    @abstractmethod
    def train(self, training_steps: int, n_train: int, n_external: int, T_max: int, seed: int = None) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Trains self for the given number of training_steps, returning the given
        number of training and external trajectories.

        Args:
            training_steps (int): How many steps to train for.
            n_train (int): Number of training trajectories to generate.
            n_external (int): Number of external trajectories to generate.
            T_max (int): Maximal number of steps in a trajectory.
            seed (int): Optional random seed for reproducibility.
        Returns:
            Tuple[List[Tuple], List[Tuple]]: Training and external trajectories,
                respectively.
        """
        pass

    @abstractmethod
    def optimal_state_val(self, state: Union[int, Tuple]) -> float:
        """
        Produces optimal Q value for a given state.
        """
        pass


class QLearnerTrainerOracle(TrainerOracle):
    """
    The QLearnerTrainerOracle is the model to conduct MIAs against. It will take
    in training trajectories and run Q learning on those. The Bellman residuals
    may then be used for MIAs.
    """

    def __init__(self, config: QLearnerTrainerOracleConfig):
        """
        Initializes a QLearnerTrainerOracle with the given config info.

        NOTE: The Data Oracle should be trained before being fed in.
        """
        super().__init__(config)
        self.data_oracle = config.data_oracle
        self.q_learner = config.q_learner

    def reset(self):
        self.q_learner.reset()

    def q_val(self, state: Union[int, Tuple], action: Union[int, Tuple]) -> float:
        return self.q_learner.q_table[state][action]

    def train(self, training_steps: int, n_train: int, n_external: int, T_max: int, seed: int = None) -> Tuple[List[Tuple], List[Tuple]]:
        trajectories = self.data_oracle.generate_trajectories(n_train + n_external, T_max, seed)
        train_trajectories = trajectories[:n_train]
        external_trajectories = trajectories[n_train:]

        if self.verbose:
            print(f"Training on {len(train_trajectories)} trajectories for {training_steps} training steps")

        for traj in train_trajectories:
            self.q_learner.replay_buffer.extend(traj)

        buffer_list = list(self.q_learner.replay_buffer)

        for _ in range(training_steps):
            transition = random.choice(buffer_list)
            self.q_learner._q_update(*transition)

        if self.verbose:
            print("Finished training")

        return train_trajectories, external_trajectories

    def optimal_state_val(self, state: Union[int, Tuple]) -> float:
        return max(self.q_learner.q_table[state].values(), default=0)

    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        return self.q_learner._select_action(state)


class DeepTrainerOracle(TrainerOracle):
    """
    Abstracts common implementation details between the Offline/Online DQN trainers.
    """

    def __init__(self, config: DeepTrainerOracleConfig):
        super().__init__(config)
        self.config = config
        self.reset()

    def reset(self):
        self.dqn = DQN(
            policy="MlpPolicy",
            env=self.config.env,
            learning_rate=self.config.learning_rate,
            learning_starts=self.config.learning_starts,
            exploration_fraction=self.config.exploration_fraction,
            exploration_final_eps=self.config.exploration_final_eps,
            batch_size=self.config.batch_size,
            buffer_size=self.config.buffer_size,
            device=self.config.device,
        )

    def _q_vals(self, state: Union[int, Tuple]):
        """
        Produces [1, n_actions] array of Q values for given state.
        """
        if isinstance(state, tuple):
            state = np.array(state)
        obs_tensor, _ = self.dqn.policy.obs_to_tensor(state)
        with torch.no_grad():
            q_values = self.dqn.policy.q_net(obs_tensor)
        return q_values

    def q_val(self, state: Union[int, Tuple], action: Union[int, Tuple]) -> float:
        return self._q_vals(state)[0, action].item()

    def optimal_state_val(self, state: Union[int, Tuple]) -> float:
        return self._q_vals(state)[0].max().item()

    def _select_action(self, state: Union[int, Tuple]) -> Union[int, Tuple]:
        if isinstance(state, tuple):
            state = np.array(state)
        action, _ = self.dqn.predict(state, deterministic=True)
        if action.ndim == 0:
            action = int(action)
        return action


class DeepOfflineTrainerOracle(DeepTrainerOracle):
    """
    A DQN-based trainer oracle with offline learning.
    """

    def __init__(self, config: DeepOfflineTrainerOracleConfig):
        super().__init__(config)
        self.data_oracle = config.data_oracle

    def train(self, training_steps: int, n_train: int, n_external: int, T_max: int, seed: int = None) -> Tuple[List[Tuple], List[Tuple]]:
        trajectories = self.data_oracle.generate_trajectories(n_train + n_external, T_max, seed)
        train_trajectories = trajectories[:n_train]
        external_trajectories = trajectories[n_train:]

        if self.verbose:
            print(f"Training on {len(train_trajectories)} trajectories for {training_steps} training steps")

        for traj in train_trajectories:
            for i, (state, action, reward, next_state) in enumerate(traj):
                done = i == len(traj) - 1
                self.dqn.replay_buffer.add(
                    np.array(state),
                    np.array(next_state),
                    np.array([action]),
                    np.array([reward]),
                    np.array([done]),
                    [{}]
                )

        self.dqn.learn(training_steps, callback=ProgressBarCallback())

        if self.verbose:
            print("Finished training")

        return train_trajectories, external_trajectories


class DeepOnlineTrainerOracle(DeepTrainerOracle):
    """
    A DQN-based trainer oracle with online learning.
    """

    def __init__(self, config: DeepOnlineTrainerOracleConfig):
        super().__init__(config)

    def train(self, training_steps: int, n_train: int, n_external: int, T_max: int, seed: int = None) -> Tuple[List[Tuple], List[Tuple]]:
        recorder = TrajectoryRecorder()
        self.dqn.learn(total_timesteps=training_steps, callback=recorder)

        external_trajectories = self.generate_trajectories(n_external, T_max, seed)

        return recorder.train_trajectories[-n_train:], external_trajectories
