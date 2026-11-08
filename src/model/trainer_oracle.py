from __future__ import annotations
import random
from typing import List, Optional, Tuple, Union, Dict, TYPE_CHECKING
from abc import ABC, abstractmethod
import math
from collections import defaultdict

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from .cql_dqn import CQLDQN
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import dp_accounting
from dp_accounting.pld import pld_privacy_accountant

if TYPE_CHECKING:
    from config import (
        TrainerOracleConfig,
        QLearnerTrainerOracleConfig,
        DeepTrainerOracleConfig,
        DeepDiscretizingTrainerOracleConfig,
        EnsemblingTrainerOracleConfig,
        DPAVIVITrainerOracleConfig,
        DPSGDTrainerOracleConfig,
    )
from .generic_model import QFunctionLearner


def ends_episode(env: gym.Env, traj: List[Tuple]) -> bool:
    """
    Whether a trajectory's last step ends its episode, so learners should not bootstrap
    from its next state. Environments with a time limit (env.unwrapped.time_limit, e.g.
    GlucoseEnv) truncate episodes that reach it: the last step is an ordinary transition and
    learners bootstrap through it. A trajectory that stops earlier was terminated.
    """
    time_limit = getattr(env.unwrapped, "time_limit", None)
    return time_limit is None or len(traj) < time_limit


class TrainerOracle(QFunctionLearner):
    """
    Abstract class for Trainer Oracle implementations, including
    scaffolding for training and Bellman residual calculations.
    """

    def __init__(self, config: TrainerOracleConfig):
        super().__init__(config.env, config.verbose, config.state_encoder)
        self.config = config

    @property
    def discount_factor(self) -> float:
        """The learner's discount factor, which the attack also uses."""
        return self.config.discount_factor

    @abstractmethod
    def train(
        self, epochs: int, n_train: int, n_external: int, T_max: int, seed: int = None
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Trains self for the given number of epochs, returning the given
        number of training and external trajectories.

        Args:
            epochs (int): How many epochs to train for. The total number of training
            steps is epochs * n_transitions, or ceil(epochs * n_transitions / batch_size)
                if batched.
            n_train (int): Number of training trajectories to generate.
            n_external (int): Number of external trajectories to generate.
            T_max (int): Maximal number of steps in a trajectory.
            seed (int): Optional random seed for reproducibility.
        Returns:
            Tuple[List[Tuple], List[Tuple]]: Training and external trajectories,
                respectively.
        """
        pass

    def reset(self):
        # Released Q-values, computed once per state and then served from here, so
        # repeated queries see the same noisy values instead of fresh draws.
        self._released: Dict[Union[int, Tuple], Dict[Union[int, Tuple], float]] = {}

    def optimal_state_val(self, state: Union[int, Tuple]) -> float:
        """
        Produces optimal Q value for a given state.
        """
        return max(self.q_vals(state).values(), default=0)

    def train_on_trajectories(self, train_trajectories: List[List[Tuple]], epochs: int):
        """
        Trains self on the given trajectories. Used by EnsemblingTrainerOracle to
        train on a disjoint shard of the data.

        Args:
            train_trajectories (List[List[Tuple]]): Trajectories to train on.
            epochs (int): How many epochs to train for.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot train on supplied trajectories"
        )

    def release_key(self, state):
        """
        Key identifying which released table entry answers queries at this state. States
        with the same key get the same released values. Defaults to the state itself.
        """
        return state

    def release(self, epsilon):
        """
        Re-releases the already trained model at a new privacy level, without retraining.
        Only possible for trainers whose training does not depend on epsilon (the noise is
        added afterwards), such as EnsemblingTrainerOracle.
        """
        raise NotImplementedError(
            f"{type(self).__name__} adds its noise during training, so it must be retrained for each epsilon"
        )

    def max_changed_entries(self, max_shard_size: int, T_max: int) -> int:
        """
        Upper bound on how many entries of this learner's released table can change when
        one of its (at most max_shard_size) training trajectories, each at most T_max
        steps, is replaced. EnsemblingTrainerOracle uses it to calibrate its noise.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no finite released table, so its sensitivity is unbounded"
        )

    def _select_action(self, state):
        """
        Greedy action, with ties broken at random. In states absent from the training data
        every action has the unseen value, and always taking the first action there can trap
        the policy (in GlucoseEnv, no insulin, which leads to ever less familiar states).
        """
        q_vals = self.q_vals(state)
        if not q_vals:
            return self.env.action_space.sample()
        best = max(q_vals.values())
        return random.choice([a for a, v in q_vals.items() if v == best])

    @abstractmethod
    def _released_q_vals(self, state) -> Dict[Union[int, Tuple], float]:
        """
        Helper to compute q values for a state once and then cache and reuse
        them after. Used by q_val and q_vals, each inference time methods.
        """
        pass

    def q_val(self, state, action) -> float:
        return self._released_q_vals(state)[action]

    def q_vals(self, state) -> Dict[Union[int, Tuple], float]:
        return dict(self._released_q_vals(state))


class QLearnerTrainerOracle(TrainerOracle):
    """
    The QLearnerTrainerOracle is a model to conduct MIAs against. It will take
    in training trajectories and run Q learning on those. The Bellman residuals
    may then be used for MIAs.
    """

    def __init__(self, config: QLearnerTrainerOracleConfig):
        """
        Initializes a QLearnerTrainerOracle with the given config info.

        NOTE: The Data Oracle should be trained before being fed in.
        """
        super().__init__(config)
        if config.discount_factor != config.q_learner.config.discount_factor:
            raise ValueError(
                f"discount_factor {config.discount_factor} must match the Q-learner's "
                f"{config.q_learner.config.discount_factor}, since the attack uses it"
            )
        self.data_oracle = config.data_oracle
        self.q_learner = config.q_learner
        self.reset()

    def reset(self):
        super().reset()
        self.observed_pairs = set()
        self.q_learner.reset()

    def _released_q_vals(self, state) -> Dict[Union[int, Tuple], float]:
        if state not in self._released:
            self._released[state] = self.q_learner.q_vals(state)
        return self._released[state]

    def max_changed_entries(self, max_shard_size: int, T_max: int) -> int:
        """
        (s, a) pairs absent from the training data are never updated, so they keep the
        unseen value (which must equal the ensemble's q_bounds[0] for this bound). Only
        pairs in the data before or after replacing one trajectory can move: at most
        (max_shard_size + 1) * T_max of them.
        """
        return (max_shard_size + 1) * T_max

    def train(
        self, epochs: int, n_train: int, n_external: int, T_max: int, seed: int = None
    ) -> Tuple[List[Tuple], List[Tuple]]:
        trajectories = self.data_oracle.generate_trajectories(
            n_train + n_external, T_max, seed
        )
        train_trajectories = trajectories[:n_train]
        external_trajectories = trajectories[n_train:]

        self.train_on_trajectories(train_trajectories, epochs)

        return train_trajectories, external_trajectories

    def train_on_trajectories(self, train_trajectories: List[List[Tuple]], epochs: int):
        if self.verbose:
            print(
                f"Training on {len(train_trajectories)} trajectories for {epochs} epochs"
            )

        for traj in train_trajectories:
            self.q_learner.replay_buffer.extend(traj)

        n_transitions = sum(len(traj) for traj in train_trajectories)
        if (
            self.q_learner.replay_buffer.maxlen is not None
            and n_transitions > self.q_learner.replay_buffer.maxlen
        ):
            raise ValueError(
                f"{n_transitions} training transitions exceed the replay buffer size "
                f"of {self.q_learner.replay_buffer.maxlen}; older member transitions would be dropped"
            )

        buffer_list = list(self.q_learner.replay_buffer)

        training_steps = epochs * len(buffer_list)

        for _ in range(training_steps):
            transition = random.choice(buffer_list)
            self.q_learner._q_update(*transition)

        if self.verbose:
            print("Finished training")


class DeepTrainerOracle(TrainerOracle):
    """
    Abstracts common implementation details between DQN trainers.
    """

    def __init__(self, config: DeepTrainerOracleConfig):
        super().__init__(config)
        self.reset()

    def reset(self):
        super().reset()
        cfg = self.config
        cql_alpha = getattr(cfg, "cql_alpha", 0.0)
        target_bounds = getattr(cfg, "target_bounds", None)
        cls = CQLDQN if cql_alpha > 0 or target_bounds is not None else DQN
        kwargs = dict(
            policy=cfg.policy,
            env=cfg.env,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            buffer_size=cfg.buffer_size,
            gamma=cfg.discount_factor,
            device=cfg.device,
        )
        if cfg.net_arch is not None:
            kwargs["policy_kwargs"] = {"net_arch": list(cfg.net_arch)}
        if cls is CQLDQN:
            kwargs["cql_alpha"] = cql_alpha
            kwargs["target_bounds"] = target_bounds
        self.dqn = cls(**kwargs)
        self.data_oracle = cfg.data_oracle

    def _raw_q_tensor(self, state: Union[int, Tuple]):
        """
        Produces [1, n_actions] tensor of Q values for given state.
        """
        if isinstance(state, tuple):
            state = np.array(state).reshape(self.env.observation_space.shape)
        obs_tensor, _ = self.dqn.policy.obs_to_tensor(state)
        with torch.no_grad():
            q_values = self.dqn.policy.q_net(obs_tensor)
        return q_values

    def _released_q_vals(self, state) -> Dict[Union[int, Tuple], float]:
        if state not in self._released:
            # Environments whose observations identify terminal states (e.g. SepsisStateVectorObs)
            # release Q = 0 there, as the tabular learner does, instead of the network's guess
            is_terminal_obs = getattr(self.env, "is_terminal_obs", None)
            if is_terminal_obs is not None and is_terminal_obs(state):
                self._released[state] = {a: 0.0 for a in range(self.env.action_space.n)}
            else:
                qv = self._raw_q_tensor(state)
                self._released[state] = {i: qv[0, i].item() for i in range(qv.shape[1])}
        return self._released[state]

    def train(
        self, epochs: int, n_train: int, n_external: int, T_max: int, seed: int = None
    ) -> Tuple[List[Tuple], List[Tuple]]:
        trajectories = self.data_oracle.generate_trajectories(
            n_train + n_external, T_max, seed
        )
        train_trajectories = trajectories[:n_train]
        external_trajectories = trajectories[n_train:]

        self.train_on_trajectories(train_trajectories, epochs)

        return train_trajectories, external_trajectories

    def train_on_trajectories(self, train_trajectories: List[List[Tuple]], epochs: int):
        if self.verbose:
            print(
                f"Training on {len(train_trajectories)} trajectories for {epochs} epochs"
            )

        obs_shape = self.dqn.observation_space.shape
        for traj in train_trajectories:
            ends = ends_episode(self.env, traj)
            for i, (state, action, reward, next_state) in enumerate(traj):
                done = ends and i == len(traj) - 1
                self.dqn.replay_buffer.add(
                    np.array(state).reshape(obs_shape),
                    np.array(next_state).reshape(obs_shape),
                    np.array([action]),
                    np.array([reward]),
                    np.array([done]),
                    [{}],
                )

        from stable_baselines3.common.utils import polyak_update
        from stable_baselines3.common.logger import configure as sb3_configure

        self.dqn.set_logger(sb3_configure(folder=None, format_strings=[]))

        total_steps = (
            math.ceil(self.dqn.replay_buffer.size() / self.config.batch_size) * epochs
        )

        interval = max(1, total_steps // self.config.target_updates)

        done = 0
        while done < total_steps:
            n = min(interval, total_steps - done)
            self.dqn.train(batch_size=self.config.batch_size, gradient_steps=n)
            done += n
            if done < total_steps:
                polyak_update(
                    self.dqn.q_net.parameters(), self.dqn.q_net_target.parameters(), 1.0
                )

        if self.verbose:
            for i in range(min(10, len(train_trajectories[0]))):
                print(self._raw_q_tensor(train_trajectories[0][i][0]))
            print("Finished training")
        self._released = {}  # values cached before training would be stale


class DeepDiscretizingTrainerOracle(DeepTrainerOracle):
    """
    Version of the deep trainer oracle that releases a Q table instead of a Q function.
    The DQN trains on the raw states, but every query is answered with the network's
    values at the nearest of a fixed, public set of points (by L2 distance), e.g. the 716
    feature vectors of SepsisStateVectorObs. Discrete state spaces use their own states as
    the points. Terminal states (when the environment can recognize them from
    observations) are released as 0. The table is released without noise; for a DP
    guarantee, use it as the sub-model of EnsemblingTrainerOracle, which clips, averages,
    and adds noise for the whole table.

    Without points on a continuous space, there is no table: every query is answered with
    the network's values at the queried observation itself. This is the undefended model
    (no discretized release), for non-private baselines only.
    """

    def __init__(self, config: DeepDiscretizingTrainerOracleConfig):
        super().__init__(config)
        cfg = self.config
        obs_space = cfg.env.observation_space
        self.n_actions = int(cfg.env.action_space.n)
        self.continuous = isinstance(obs_space, gym.spaces.Box)
        if self.continuous and cfg.points is not None:
            self.points = np.asarray(cfg.points, dtype=float)
            self.n_cells = len(self.points)
        else:
            self.points = None
            self.n_cells = None if self.continuous else int(obs_space.n)

    def release_key(self, state):
        """
        The point nearest to the state by L2 distance (the state itself for discrete spaces,
        and for continuous spaces without points).
        """
        if self.points is None:
            return tuple(np.asarray(state, dtype=float).flatten()) if self.continuous else state
        arr = np.asarray(state, dtype=float).flatten()
        idx = np.sqrt(np.sum((arr - self.points) ** 2, axis=1)).argmin()
        return tuple(self.points[idx].flatten())

    def max_changed_entries(self, max_shard_size: int, T_max: int) -> int:
        """Retraining the network can move every entry of the table."""
        if self.n_cells is None:
            raise ValueError(
                "Without `points`, the released Q function is not a finite table, so its sensitivity is unbounded"
            )
        return self.n_cells * self.n_actions

    def _released_q_vals(self, state) -> Dict[Union[int, Tuple], float]:
        cell = self.release_key(state)
        if cell not in self._released:
            is_terminal_obs = getattr(self.env, "is_terminal_obs", None)
            if is_terminal_obs is not None and is_terminal_obs(cell):
                self._released[cell] = {a: 0.0 for a in range(self.n_actions)}
            else:
                qv = self._raw_q_tensor(cell)[0].cpu().numpy()
                self._released[cell] = {a: float(v) for a, v in enumerate(qv)}

        # print(max(self._released[cell].values()))
        return self._released[cell]


class DPSGDTrainerOracle(TrainerOracle):
    """
    Offline DQN (optionally CQL) trained with DP-SGD through Opacus, with one trajectory
    as the privacy unit. Each dataset row is one trajectory, padded to the longest
    training trajectory with a mask. A row's loss is the sum (or mean) over its real steps
    of the Huber TD loss, plus cql_alpha times the CQL penalty.
    """

    def __init__(self, config: DPSGDTrainerOracleConfig):
        super().__init__(config)
        self.data_oracle = config.data_oracle
        obs_space = config.env.observation_space
        if not isinstance(config.env.action_space, gym.spaces.Discrete):
            raise ValueError("DPSGDTrainerOracle requires a discrete action space")
        self.n_actions = int(config.env.action_space.n)
        self.discrete_states = isinstance(obs_space, gym.spaces.Discrete)
        self.input_dim = (
            int(obs_space.n) if self.discrete_states else int(np.prod(obs_space.shape))
        )
        # A wrapper that recognizes terminal states from observations (e.g. SepsisStateVectorObs)
        # takes precedence over the base environment's check on raw states
        self._is_terminal = getattr(config.env, "is_terminal_obs", None) or getattr(
            config.env.unwrapped, "_is_terminal_state", None
        )
        self._release_zero_at_terminal = hasattr(config.env, "is_terminal_obs")
        self.device = torch.device(config.device)
        self.reset()

    def reset(self):
        super().reset()
        self.q_net = self._build_q_net()
        self.target_net = self._build_q_net()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.privacy_report = {}

    def _build_q_net(self) -> nn.Module:
        layers, width = [], self.input_dim
        for hidden in self.config.hidden_sizes:
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, self.n_actions))
        return nn.Sequential(*layers).to(self.device)

    def _state_array(self, states) -> np.ndarray:
        if self.discrete_states:
            return np.asarray(states, dtype=np.int64)
        return np.asarray(
            [np.asarray(s, dtype=np.float32).flatten() for s in states],
            dtype=np.float32,
        )

    def _network_input(self, states: torch.Tensor) -> torch.Tensor:
        """One-hot encodes discrete states; continuous states are used as they are."""
        if self.discrete_states:
            return F.one_hot(states, self.input_dim).float()
        return states

    def _padded_dataset(self, trajectories: List[List[Tuple]]) -> TensorDataset:
        """
        One row per trajectory, padded to the longest: states, actions, rewards, next
        states, done flags (terminal next state), and a mask of real steps.
        """
        n, length = len(trajectories), max(len(traj) for traj in trajectories)
        state_shape = (
            (n, length) if self.discrete_states else (n, length, self.input_dim)
        )
        states = np.zeros(
            state_shape, dtype=np.int64 if self.discrete_states else np.float32
        )
        next_states = np.zeros_like(states)
        actions = np.zeros((n, length), dtype=np.int64)
        rewards = np.zeros((n, length), dtype=np.float32)
        dones = np.zeros((n, length), dtype=np.float32)
        mask = np.zeros((n, length), dtype=np.float32)
        for i, traj in enumerate(trajectories):
            t = len(traj)
            states[i, :t] = self._state_array([step[0] for step in traj])
            next_states[i, :t] = self._state_array([step[3] for step in traj])
            actions[i, :t] = [step[1] for step in traj]
            rewards[i, :t] = [step[2] for step in traj]
            if self._is_terminal is not None:
                dones[i, :t] = [float(self._is_terminal(step[3])) for step in traj]
            else:
                dones[i, t - 1] = float(ends_episode(self.config.env, traj))
            mask[i, :t] = 1.0
        return TensorDataset(
            *(
                torch.from_numpy(x)
                for x in (states, actions, rewards, next_states, dones, mask)
            )
        )

    def _trajectory_losses(self, model: nn.Module, batch) -> torch.Tensor:
        cfg = self.config
        states, actions, rewards, next_states, dones, mask = (
            x.to(self.device) for x in batch
        )
        q_all = model(self._network_input(states))  # (B, L, A)
        q_taken = q_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # (B, L)
        with torch.no_grad():
            next_values = (
                self.target_net(self._network_input(next_states)).max(dim=-1).values
            )
            targets = rewards + cfg.discount_factor * (1 - dones) * next_values
            if cfg.target_bounds is not None:
                targets = targets.clamp(*cfg.target_bounds)
        step_losses = F.smooth_l1_loss(q_taken, targets, reduction="none")
        if cfg.cql_alpha > 0:
            step_losses = step_losses + cfg.cql_alpha * (
                torch.logsumexp(q_all, dim=-1) - q_taken
            )
        step_losses = step_losses * mask
        if cfg.trajectory_loss == "sum":
            return step_losses.sum(dim=1)
        if cfg.trajectory_loss == "mean":
            return step_losses.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        raise ValueError(f"Unknown trajectory_loss {cfg.trajectory_loss}")

    def train(
        self, epochs: int, n_train: int, n_external: int, T_max: int, seed: int = None
    ) -> Tuple[List[Tuple], List[Tuple]]:
        trajectories = self.data_oracle.generate_trajectories(
            n_train + n_external, T_max, seed
        )
        train_trajectories = trajectories[:n_train]
        external_trajectories = trajectories[n_train:]
        self.train_on_trajectories(train_trajectories, epochs)
        return train_trajectories, external_trajectories

    def train_on_trajectories(self, train_trajectories: List[List[Tuple]], epochs: int):
        cfg = self.config
        loader = DataLoader(
            self._padded_dataset(train_trajectories),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.learning_rate)
        model = self.q_net
        engine = None
        if cfg.epsilon is not None:
            engine = PrivacyEngine(accountant=cfg.accountant)
            if cfg.neighboring_relation == "replace_one":
                # Same sample rate and step count Opacus would use: q = 1 / len(loader), T = epochs / q
                sample_rate, planned_steps = 1 / len(loader), epochs * len(loader)
                noise_multiplier = replace_one_noise_multiplier(
                    cfg.epsilon, cfg.delta, sample_rate, planned_steps
                )
                model, optimizer, loader = engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=loader,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=cfg.max_grad_norm,
                )
            elif cfg.neighboring_relation == "add_or_remove_one":
                model, optimizer, loader = engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=loader,
                    target_epsilon=cfg.epsilon,
                    target_delta=cfg.delta,
                    epochs=epochs,
                    max_grad_norm=cfg.max_grad_norm,
                )
            else:
                raise ValueError(
                    f"Unknown neighboring_relation {cfg.neighboring_relation}"
                )

        self.q_net.train()
        # Copies are scheduled by planned batch, so every run makes the same number of
        # target updates (target_updates - 1, since a copy after the last batch would never be used)
        planned = epochs * len(loader)
        interval = max(1, planned // cfg.target_updates)
        steps = batches = 0
        for _ in range(epochs):
            for batch in loader:
                # Poisson sampling can draw an empty batch; skipping it only lowers the privacy spent
                if len(batch[0]) > 0:
                    loss = self._trajectory_losses(model, batch).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    steps += 1
                batches += 1
                if batches % interval == 0 and batches < planned:
                    self.target_net.load_state_dict(self.q_net.state_dict())
        self.q_net.eval()

        self.privacy_report = {
            "epsilon_target": cfg.epsilon,
            "delta": cfg.delta,
            "steps": steps,
            "max_grad_norm": cfg.max_grad_norm,
        }
        if engine is not None:
            sample_rate = float(1 / len(loader))
            if cfg.neighboring_relation == "replace_one":
                interval = _pld_interval(cfg.epsilon)
                epsilon_spent = replace_one_epsilon(
                    optimizer.noise_multiplier, sample_rate, steps, cfg.delta, interval
                )
                # Reference value from the same PLD accountant; Opacus's PRV accountant needs
                # ~5 GB at small noise multipliers (large epsilon) and then returns inf
                acc = _replace_one_accountant(dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE, interval)
                acc.compose(_dpsgd_event(optimizer.noise_multiplier, sample_rate, steps))
                add_remove_epsilon = float(acc.get_epsilon(cfg.delta)) if steps else 0.0
            else:
                add_remove_epsilon = float(engine.get_epsilon(cfg.delta))  # Opacus's accountant
                epsilon_spent = add_remove_epsilon
            self.privacy_report.update(
                {
                    "neighboring_relation": cfg.neighboring_relation,
                    "epsilon_spent": epsilon_spent,  # the guarantee, under neighboring_relation
                    "epsilon_add_remove": add_remove_epsilon,  # same run under add/remove-one, for reference
                    "noise_multiplier": float(optimizer.noise_multiplier),
                    "sample_rate": sample_rate,
                    "accountant": cfg.accountant,
                }
            )
            model.to_standard_module()  # removes Opacus's hooks; self.q_net keeps the trained weights
        if self.verbose:
            print(f"DP-SGD: {self.privacy_report}")
        self._released = {}  # values cached before training would be stale

    def _released_q_vals(self, state) -> Dict[Union[int, Tuple], float]:
        if (
            state not in self._released
            and self._release_zero_at_terminal
            and self._is_terminal(state)
        ):
            self._released[state] = {a: 0.0 for a in range(self.n_actions)}
        if state not in self._released:
            inputs = torch.from_numpy(self._state_array([state])).to(self.device)
            with torch.no_grad():
                qv = self.q_net(self._network_input(inputs))[0].cpu().numpy()
            self._released[state] = {a: float(v) for a, v in enumerate(qv)}
        return self._released[state]


class EnsemblingTrainerOracle(TrainerOracle):
    """
    Composes with k trainer oracles trained on disjoint shards of the training
    trajectories (subsample and aggregate). Each released Q-value is the mean of the
    sub-models' values clipped to q_bounds, plus Gaussian noise giving one
    (epsilon, delta)-DP guarantee for the whole released table.
    """

    def __init__(self, config: EnsemblingTrainerOracleConfig):
        super().__init__(config)
        self.data_oracle = config.data_oracle
        self.epsilon = config.epsilon  # privacy level of the current release; see release()
        self.reset()

    def reset(self):
        super().reset()
        cfg = self.config
        self.trainer_oracles = [cfg.trainer_factory() for _ in range(cfg.k)]
        for t in self.trainer_oracles:
            t.reset()
        self._released: Dict[Union[int, Tuple], Dict[Union[int, Tuple], float]] = {}

    def _clip(self, val):
        return np.clip(val, self.config.q_bounds[0], self.config.q_bounds[1]).item()

    def _table_noise_std(self, n_train: int, T_max: int) -> float:
        """Gaussian std for (epsilon, delta)-DP of the whole released table (see class docstring)."""
        cfg = self.config
        lo, hi = cfg.q_bounds
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(
                f"q_bounds {cfg.q_bounds} must be finite to calibrate noise (epsilon={self.epsilon}); use epsilon=None for no noise"
            )
        m = int(np.ceil(n_train / cfg.k))
        max_changed = self.trainer_oracles[0].max_changed_entries(m, T_max)
        sensitivity = (hi - lo) * np.sqrt(max_changed) / cfg.k
        log_term = np.log(1 / cfg.delta)
        rho = (np.sqrt(log_term + self.epsilon) - np.sqrt(log_term)) ** 2
        std = float(sensitivity / np.sqrt(2 * rho))
        self.noise_info = {
            "m": m,
            "max_changed_pairs": max_changed,
            "l2_sensitivity": float(sensitivity),
            "rho": float(rho),
            "epsilon": self.epsilon,
            "delta": cfg.delta,
            "noise_std": std,
        }
        return std

    def _aggregate(self, vals) -> float:
        """
        Mean of the k clipped values plus Gaussian noise with the whole-table std computed
        in train(). epsilon=None releases the mean without noise (no formal guarantee).
        """
        mean = float(np.mean(vals))
        if self.epsilon is None:
            return mean
        return mean + np.random.normal(0.0, self._noise_std)

    def _released_q_vals(self, state) -> Dict[Union[int, Tuple], float]:
        key = self.trainer_oracles[0].release_key(state)
        if key not in self._released:
            per_model = [t.q_vals(state) for t in self.trainer_oracles]
            lo = self.config.q_bounds[0]
            self._released[key] = {
                a: self._aggregate([self._clip(qv.get(a, lo)) for qv in per_model])
                for a in self.actions
            }
        return self._released[key]

    def release(self, epsilon):
        """
        Releases the trained ensemble at the given epsilon. Training does not depend on
        epsilon, only the noise on the averaged table does, so this redraws the noise
        (lazily, per entry) at the new level and discards previously released values.
        """
        self.epsilon = epsilon
        if epsilon is not None:
            self._noise_std = self._table_noise_std(self._n_train, self._T_max)
        self._released = {}

    def train(
        self, epochs: int, n_train: int, n_external: int, T_max: int, seed: int = None
    ) -> Tuple[List[Tuple], List[Tuple]]:
        k = self.config.k
        if n_train < k:
            raise ValueError(
                f"Need at least k={k} training trajectories, got {n_train}"
            )

        self._n_train, self._T_max = n_train, T_max
        self.release(self.epsilon)

        # Generate once and partition, so each trajectory trains exactly one oracle
        trajectories = self.data_oracle.generate_trajectories(
            n_train + n_external, T_max, seed
        )
        train_trajectories = trajectories[:n_train]
        external_trajectories = trajectories[n_train:]

        start = 0
        for i, trainer_oracle in enumerate(self.trainer_oracles):
            shard_size = n_train // k + (i < n_train % k)
            trainer_oracle.train_on_trajectories(
                train_trajectories[start : start + shard_size], epochs
            )
            start += shard_size

        return train_trajectories, external_trajectories


class DPAVIVITrainerOracle(TrainerOracle):
    """
    DP-APVI (Qiao & Wang 2022, Algorithm 1), adapted to a stationary MDP: noisy counts
    pooled over steps, then T_max rounds of pessimistic value iteration directly on the
    private model. H in the paper is T_max here.
    """

    def __init__(self, config: DPAVIVITrainerOracleConfig):
        super().__init__(config)
        self.data_oracle = config.data_oracle
        if not (
            isinstance(self.env.observation_space, gym.spaces.Discrete)
            and isinstance(self.env.action_space, gym.spaces.Discrete)
        ):
            raise ValueError(
                "DPAVIVITrainerOracle requires discrete states and actions"
            )
        self.n_states = int(self.env.observation_space.n)
        self.n_actions = int(self.env.action_space.n)
        self.reset()

    def reset(self):
        super().reset()
        self.privacy_report = {}
        self.q_table = np.zeros((self.n_states, self.n_actions))

    def noise_std(self) -> float:
        """
        Gaussian std for (epsilon, delta)-DP via rho-zCDP.
        """
        cfg = self.config
        if cfg.epsilon is None:
            return 0.0
        log_term = np.log(1 / cfg.delta)
        rho = (np.sqrt(log_term + cfg.epsilon) - np.sqrt(log_term)) ** 2
        return np.sqrt(4 * cfg.T_max) / np.sqrt(2 * rho)

    def _noise_bound(self) -> float:
        """
        E (Lemma C.2): with probability 1 - failure_prob, every count's noise is at most E / 2
        in absolute value. E_rho = 2 sqrt(2) sigma sqrt(log(4HS^2A / delta)), which equals
        4 sqrt(H log(4HS^2A / delta) / rho) for sigma^2 = 2H / rho.
        """
        cfg = self.config
        log_term = np.log(
            4 * cfg.T_max * self.n_states**2 * self.n_actions / cfg.failure_prob
        )
        return 2 * np.sqrt(2) * self.noise_std() * np.sqrt(log_term)

    def _noisy_counts(
        self, member_trajectories: List[List[Tuple]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Noisy (n_sa, n_sas), before clipping. Only the first visit to each (s, a) in a
        trajectory is counted, with its next state, so each trajectory adds at most 1 to
        any cell and the true counts satisfy sum over s' of n(s, a, s') = n(s, a), as the
        paper's per-step counts do.
        """
        cfg = self.config
        S, A = self.n_states, self.n_actions
        n_sa = np.zeros((S, A))
        n_sas = np.zeros((S, A, S), dtype=np.float32)
        for traj in member_trajectories:
            first_visits = {}
            for s, a, _, s2 in traj[: cfg.T_max]:
                if (s, a) not in first_visits:
                    first_visits[(s, a)] = s2
            for (s, a), s2 in first_visits.items():
                n_sa[s, a] += 1
                n_sas[s, a, s2] += 1

        std = self.noise_std()
        n_sa = n_sa + np.random.normal(0.0, std, n_sa.shape)
        noise = np.random.normal(0.0, std, n_sas.shape).astype(np.float32)
        if cfg.public_support is not None:
            # Cells outside the public support are known to be impossible and released as 0
            noise *= cfg.public_support
            n_sas *= cfg.public_support
        n_sas += noise

        self.privacy_report = {
            "epsilon": cfg.epsilon,
            "delta": cfg.delta,
            "noise_std": float(std),
            "T_max": cfg.T_max,
        }
        return n_sa, n_sas

    def _consistent_counts(
        self, n_sa: np.ndarray, n_sas: np.ndarray, E: float
    ) -> np.ndarray:
        if self.config.consistency_solver == "linprog":
            return consistent_counts_linprog(n_sa, n_sas, E, self.config.public_support)
        if self.config.consistency_solver == "closed_form":
            return consistent_counts_closed_form(
                n_sa, n_sas, E, self.config.public_support
            )
        raise ValueError(f"Unknown consistency_solver {self.config.consistency_solver}")

    def _released_q_vals(self, state) -> Dict[Union[int, Tuple], float]:
        if state not in self._released:
            self._released[state] = {
                a: float(v) for a, v in enumerate(self.q_table[state])
            }
        return self._released[state]

    def train(
        self, epochs: int, n_train: int, n_external: int, T_max: int, seed: int = None
    ) -> Tuple[List[Tuple], List[Tuple]]:
        trajectories = self.data_oracle.generate_trajectories(
            n_train + n_external, T_max, seed
        )
        train_trajectories = trajectories[:n_train]
        external_trajectories = trajectories[n_train:]

        cfg = self.config
        S, A = self.n_states, self.n_actions
        H = cfg.value_bound if cfg.value_bound is not None else cfg.T_max

        C_1 = np.sqrt(2)
        C_2 = 16
        C = cfg.C
        E = self._noise_bound()
        iota = np.log(cfg.T_max * S * A / cfg.failure_prob)

        n_sa, n_sas = self._noisy_counts(train_trajectories)
        n_sa = np.maximum(n_sa, 0.0)  # Eq. 2's {.}^+
        n_sas = np.maximum(n_sas, 0.0)
        x = self._consistent_counts(n_sa, n_sas, E)  # S * A * S
        n = x.sum(axis=2)  # S * A, n~(s, a)
        known = n > E  # S * A
        prob_matrix = np.where(
            known[..., None], x / np.maximum(n, 1e-12)[..., None], 1 / S
        )  # S * A * S
        del x

        # The algorithm assumes a known, public reward; only the transitions are private
        R = cfg.reward_table if cfg.reward_table is not None else self.env.unwrapped._r_mat
        R = np.asarray(R)
        if R.ndim == 2:  # S * S': the same reward for every action
            R = R[:, None, :]
        expected_reward = (prob_matrix * R).sum(axis=2)  # S * A

        V = np.zeros(S)
        for _ in range(cfg.T_max):
            EV, EV2 = prob_matrix @ V, prob_matrix @ (V**2)  # S * A
            var = np.maximum(EV2 - EV**2, 0)
            Gamma = np.where(
                known,
                cfg.penalty_scale
                * (
                    C_1 * np.sqrt(var * iota / np.maximum(n - E, 1e-12))
                    + C_2 * S * H * E * iota / np.maximum(n, 1e-12)
                ),
                C * H,
            )
            self.q_table = np.clip(expected_reward + EV - Gamma, 0, H)
            V = np.max(self.q_table, axis=1)

        self._released = {}  # values cached before training would be stale
        self.privacy_report.update({"E": float(E), "known_pairs": int(known.sum())})
        if self.verbose:
            print(f"DP-APVI: {self.privacy_report}")

        # The members are the real training trajectories; the model never saw them directly
        return train_trajectories, external_trajectories


def consistent_counts_linprog(
    n_sa: np.ndarray, n_sas: np.ndarray, E: float, support: np.ndarray = None
) -> np.ndarray:
    """
    Qiao & Wang 2023, Eq. 3, solved with scipy's LP solver for each (s, a). With
    c = n'(s, a, .), the variables are (x, t):
        minimize t
        s.t.  x_s' - t <= c_s'          (x - c <= t)
             -x_s' - t <= -c_s'         (c - x <= t)
              sum(x)   <= n'(s, a) + E / 2
             -sum(x)   <= -(n'(s, a) - E / 2)
              x >= 0, t >= 0
    Cells outside the public support (if given) have c = 0 and are fixed at 0. Slow.
    """
    S, A = n_sa.shape
    n_next = n_sas.shape[2]
    identity = sparse.eye(n_next, format="csr")
    ones_col = sparse.csr_matrix(np.ones((n_next, 1)))
    sum_row = sparse.csr_matrix(np.r_[np.ones(n_next), 0.0][None])
    A_ub = sparse.vstack(
        [
            sparse.hstack([identity, -ones_col]),
            sparse.hstack([-identity, -ones_col]),
            sum_row,
            -sum_row,
        ]
    ).tocsr()
    cost = np.r_[np.zeros(n_next), 1.0]

    x = np.zeros(n_sas.shape)
    for s in range(S):
        for a in range(A):
            c = n_sas[s, a].astype(np.float64)
            if support is not None:
                c = c * support[s, a]
            b_ub = np.r_[c, -c, n_sa[s, a] + E / 2, -(n_sa[s, a] - E / 2)]
            bounds = (
                [(0, None)] * n_next
                if support is None
                else [(0, None if support[s, a, s2] else 0) for s2 in range(n_next)]
            )
            result = linprog(
                cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds + [(0, None)], method="highs"
            )
            if not result.success:
                raise RuntimeError(
                    f"Consistency LP failed at (s={s}, a={a}): {result.message}"
                )
            x[s, a] = np.maximum(
                result.x[:n_next], 0.0
            )  # remove solver round-off below 0
    return x


def consistent_counts_closed_form(
    n_sa: np.ndarray,
    n_sas: np.ndarray,
    E: float,
    support: np.ndarray = None,
    n_iters: int = 60,
) -> np.ndarray:
    """
    Same problem as consistent_counts_linprog, solved directly for all (s, a) at once.
    For one row, write c = n'(s, a, .) >= 0 (already clipped), lo = n'(s, a) - E / 2,
    hi = n'(s, a) + E / 2, and r = max_s' |x_s' - c_s'| for the objective. There are
    three cases:

    1. lo <= sum(c) <= hi: x = c is feasible with r = 0, so it is optimal.

    2. sum(c) > hi (too much mass). Let f(t) = sum(max(c - t, 0)), which falls
       continuously from sum(c) at t = 0 to 0 at t = max(c), and let t* solve f(t*) = hi.
       - Lower bound: any feasible x with objective r has x >= c - r and x >= 0, so
         sum(x) >= f(r). Feasibility needs sum(x) <= hi = f(t*), and f is decreasing,
         so r >= t*.
       - Attained: x = max(c - t*, 0) has sum(x) = hi, x >= 0 and |x - c| <= t*.
       So x = max(c - t*, 0) is optimal. t* is found by bisection.

    3. sum(c) < lo (too little mass). Any x with objective r has x <= c + r, so
       sum(x) <= sum(c) + S * r, and feasibility needs r >= (lo - sum(c)) / S. Adding
       that amount to every cell attains it. With a public support, only the k cells
       in the support can move, and the same argument gives (lo - sum(c)) / k.
    """
    c = n_sas.astype(np.float64)
    if support is not None:
        c = c * support
    lo = n_sa - E / 2
    hi = n_sa + E / 2
    total = c.sum(axis=2)  # S * A totals

    # Case 2: bisection on t for f(t) = hi, run on every row (only used where total > hi)
    t_low = np.zeros(n_sa.shape)
    t_high = c.max(axis=2)
    for _ in range(n_iters):
        t_mid = (t_low + t_high) / 2
        too_much = np.maximum(c - t_mid[..., None], 0.0).sum(axis=2) > hi
        t_low = np.where(too_much, t_mid, t_low)
        t_high = np.where(too_much, t_high, t_mid)
    x_down = np.maximum(c - t_high[..., None], 0.0)

    # Case 3: shift every cell (in the support) up by the same amount
    n_free = (
        np.full(n_sa.shape, c.shape[2])
        if support is None
        else np.maximum(support.sum(axis=2), 1)
    )
    shift = (lo - total) / n_free
    x_up = c + shift[..., None] * (1.0 if support is None else support)

    return np.where(
        (total > hi)[..., None], x_down, np.where((total < lo)[..., None], x_up, c)
    )


# Largest target epsilon the PLD accountant can calibrate without excessive memory
MAX_PLD_EPSILON = 3e5


def _pld_interval(epsilon: float) -> float:
    """
    Discretization interval for the PLD accountant at this epsilon. The accountant's memory
    grows roughly with epsilon / interval: the default interval (1e-4) needs ~6 GB at
    epsilon 3000. Scaling the interval with epsilon keeps each calibration at ~1-1.6 GB and
    gives the same noise multipliers to 4 digits; above 0.1 the discretization is too coarse.
    Coarser intervals only overestimate epsilon (the estimate is pessimistic), so it remains
    an upper bound.
    """
    return min(0.1, max(1e-3, epsilon / 3e6))


def _replace_one_accountant(
    neighboring_relation=dp_accounting.NeighboringRelation.REPLACE_ONE,
    interval: float = 1e-3,
) -> pld_privacy_accountant.PLDAccountant:
    return pld_privacy_accountant.PLDAccountant(
        neighboring_relation=neighboring_relation,
        value_discretization_interval=interval,
    )


def _dpsgd_event(
    noise_multiplier: float, sample_rate: float, steps: int
) -> dp_accounting.DpEvent:
    """steps Poisson-sampled Gaussian steps, with noise std noise_multiplier * C on a sum of norm-C clipped gradients."""
    return dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sample_rate, dp_accounting.GaussianDpEvent(noise_multiplier)
        ),
        steps,
    )


def replace_one_epsilon(
    noise_multiplier: float, sample_rate: float, steps: int, delta: float, interval: float = 1e-3
) -> float:
    """
    Epsilon of DP-SGD at the given delta under replace-one neighbours, from dp_accounting's
    PLD accountant. With independent Poisson coins, the replaced and replacing examples can
    share one coin, so each step compares (1 - q) N(0, s^2) + q N(+1, s^2) against
    (1 - q) N(0, s^2) + q N(-1, s^2) (in units of C), which is the pair the accountant
    uses for REPLACE_ONE.
    """
    if steps == 0:
        return 0.0
    accountant = _replace_one_accountant(interval=interval)
    accountant.compose(_dpsgd_event(noise_multiplier, sample_rate, steps))
    return float(accountant.get_epsilon(delta))


def replace_one_noise_multiplier(
    target_epsilon: float, target_delta: float, sample_rate: float, steps: int
) -> float:
    """
    Smallest noise multiplier (up to calibration tolerance) for which `steps` DP-SGD steps
    at this sample rate are (target_epsilon, target_delta)-DP under replace-one neighbours.
    """
    if target_epsilon > MAX_PLD_EPSILON:
        raise ValueError(
            f"Target epsilon {target_epsilon:g} is above {MAX_PLD_EPSILON:g}, where calibrating the "
            "noise needs too much memory (see _pld_interval); use epsilon=None for no noise"
        )
    interval = _pld_interval(target_epsilon)
    sigma = dp_accounting.calibrate_dp_mechanism(
        lambda: _replace_one_accountant(interval=interval),
        lambda noise_multiplier: _dpsgd_event(noise_multiplier, sample_rate, steps),
        target_epsilon,
        target_delta,
    )
    # The calibration returns a value within its tolerance of the optimum; make sure it is on the private side
    while replace_one_epsilon(sigma, sample_rate, steps, target_delta, interval) > target_epsilon:
        sigma *= 1 + 1e-4
    return float(sigma)
