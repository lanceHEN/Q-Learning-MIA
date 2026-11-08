"""
Builds the experiments in the paper: each environment's settings and behavior policy, and
one config builder per trainer oracle (the ensemble, DP-APVI, and DP-SGD). Builders take
every setting as an argument, so a functools.partial of one can be sent to run_sweep's
worker processes, which re-import this module.
"""
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import icu_sepsis  # registers Sepsis/ICU-Sepsis-v2 with gymnasium
import numpy as np

from config import (
    DeepDiscretizingTrainerOracleConfig,
    DPAVIVITrainerOracleConfig,
    DPSGDTrainerOracleConfig,
    EnsemblingTrainerOracleConfig,
    ExperimentRunnerConfig,
    QLearnerConfig,
    QLearnerDataOracleConfig,
    QLearnerTrainerOracleConfig,
)
from src.model import (
    ClusteringMIAClassifier,
    DeepDiscretizingTrainerOracle,
    DiscretizedPolicyDataOracle,
    DPAVIVITrainerOracle,
    DPSGDTrainerOracle,
    EnsemblingTrainerOracle,
    FeatureMappedDataOracle,
    GlucoseDiscreteObs,
    GlucoseEnv,
    QLearner,
    QLearnerDataOracle,
    QLearnerTrainerOracle,
    SepsisStateVectorObs,
)

ENVS = ("sepsis", "glucose")

# Episode length; glucose episodes are one day of 48 half-hour steps
T_MAX = {"sepsis": 50, "glucose": 48}

# Discount of the trainer oracles (and the attack, which reads it from them). ICU-Sepsis episodes
# end in outcome states, so returns are undiscounted. Glucose is a continuing task cut after a day
# (a truncation, which learners bootstrap through): 47/48 gives an effective horizon of one day and
# keeps values in [0, 1], since rewards are at most 1/48 and (1/48) / (1 - 47/48) = 1.
GAMMA = {"sepsis": 1.0, "glucose": 47 / 48}

# Behavior policy (data oracle) training schedule and softmax temperature tau.
# Sepsis: epsilon starts above 1, so actions are uniformly random for the first ln(10) / 9e-7 = 2.56M
# steps, after which epsilon decays geometrically from 1 to 0.11. Greedy return 0.826; tau = 0.01 takes
# the greedy action 71% of the time, with return 0.816. (Glucose's schedule, epsilon = 1 throughout
# with a decaying alpha, models/sepsis_qlearner_data_oracle_C.pkl, gave similar returns but data the
# CQL ensemble learned less from: 0.802 vs 0.815 at k = 20, N = 1000.)
DATA_ORACLE = {
    "sepsis": dict(steps=5_000_000, alpha=0.1, alpha_decay_rate=1.0, epsilon=10, decay_rate=0.9999991, tau=0.01),
    "glucose": dict(steps=10_000_000, alpha=0.03, alpha_decay_rate=(0.001 / 0.03) ** (1 / 10_000_000),
                    epsilon=1, decay_rate=1.0, tau=0.002),
}

# Privacy levels shared by every sweep, so methods and environments are compared at the same
# epsilons. All mechanisms are (epsilon, 1e-5)-DP for one trajectory under replace-one
# neighbours; None is the non-private reference. DP-SGD's accountant handles epsilon <= 3e5.
EPSILONS = [1, 10, 100, 1000, 10_000, 100_000, None]
# Dataset sizes (members + non-members; half are members) shared by every sweep
N_TRAJECTORIES = [100, 200, 500, 1000]
# Ensemble sizes for EnsemblingTrainerOracle, shared by every environment. k is at most the
# number of members, so 50 is the largest that works at N = 100 (one trajectory per model)
K_VALUES = [1, 5, 10, 20, 50]
# Training epochs of each kind of learner
EPOCHS = {"tabular": 10_000, "cql": 300, "dpsgd": 300}

# Settings shared by every experiment's ExperimentRunnerConfig
_RUNNER_DEFAULTS = dict(
    seed=None,
    n_trajectories=500,  # overridden by the sweep
    train_external_split=0.5,
    mia_train_test_split=0.5,
    n_eval_trajectories=200,
    save_transition_data=False,
    save_plots=False,  # sweeps write one raw results file per run and no plots
)


def data_oracle_path(env_name: str) -> str:
    return f"models/{env_name}_qlearner_data_oracle.pkl"


def make_tabular_env(env_name: str, end_on_truncation: bool = False) -> gym.Env:
    """
    The environment with integer states, which the data oracle's Q-table is indexed by.
    end_on_truncation ends each glucose day in GlucoseDiscreteObs.end_state, as when the saved
    glucose data oracle was trained (see GlucoseDiscreteObs); only its training uses it.
    """
    if env_name == "sepsis":
        return gym.make("Sepsis/ICU-Sepsis-v2")
    if env_name == "glucose":
        return GlucoseDiscreteObs(GlucoseEnv(), end_on_truncation=end_on_truncation)
    raise ValueError(f"Unknown environment {env_name}")


def make_data_oracle(env_name: str, env: gym.Env) -> QLearnerDataOracle:
    """The (untrained) behavior policy for env_name, on its integer states (see make_tabular_env)."""
    spec = DATA_ORACLE[env_name]
    q_learner = QLearner(
        QLearnerConfig(
            env=env,
            alpha=spec["alpha"],
            epsilon=spec["epsilon"],
            buffer_size=1,
            discount_factor=1,
        )
    )
    return QLearnerDataOracle(
        QLearnerDataOracleConfig(
            env=env,
            decay_rate=spec["decay_rate"],
            alpha_decay_rate=spec["alpha_decay_rate"],
            learning_starts=10000,
            q_learner=q_learner,
            tau=spec["tau"],
        )
    )


def load_or_train_data_oracle(env_name: str) -> QLearnerDataOracle:
    """
    The behavior policy saved at data_oracle_path(env_name), training and saving it first if
    there is none. It is trained with each glucose day ending in end_state and discount 1, as
    the saved oracles were.
    """
    path = data_oracle_path(env_name)
    if not Path(path).exists():
        training_oracle = make_data_oracle(env_name, make_tabular_env(env_name, end_on_truncation=True))
        training_oracle.train(DATA_ORACLE[env_name]["steps"])
        training_oracle.save(path)
    data_oracle = make_data_oracle(env_name, make_tabular_env(env_name))
    data_oracle.load(path)
    return data_oracle


def make_env_and_data_oracle(env_name: str, discrete: bool) -> Tuple[gym.Env, object]:
    """
    The environment for a trainer oracle, and a data oracle generating its trajectories with
    the saved behavior policy (so every repeat shares it).
    discrete=True gives integer states (tabular learners, DP-APVI); discrete=False gives
    feature vectors (deep learners, DP-SGD), with env.features the public points to release Q
    values for. The behavior policy acts on integer states either way: for Sepsis its integer
    states are mapped to feature vectors, for glucose each observation is mapped to its grid
    state to choose the action, and the trajectories keep the observations.
    """
    tabular_env = make_tabular_env(env_name)
    behavior = make_data_oracle(env_name, tabular_env)
    behavior.load(data_oracle_path(env_name))
    if discrete:
        return tabular_env, behavior
    if env_name == "sepsis":
        env = SepsisStateVectorObs(gym.make("Sepsis/ICU-Sepsis-v2"))
        return env, FeatureMappedDataOracle(behavior, env.encode)
    env = GlucoseEnv()
    return env, DiscretizedPolicyDataOracle(behavior, env, env.grid_index)


def _runner_config(name: str, env_name: str, env: gym.Env, trainer_oracle, epochs: int) -> ExperimentRunnerConfig:
    return ExperimentRunnerConfig(
        experiment_name=name,
        env=env,
        trainer_oracle=trainer_oracle,
        mia_classifier=ClusteringMIAClassifier(trainer_oracle),
        T_max=T_MAX[env_name],
        trainer_oracle_epochs=epochs,
        **_RUNNER_DEFAULTS,
    )


# ---------- Subsample-and-aggregate ensemble (also the non-private baselines, with k = 1) ----------

def build_ensemble_config(
    ensemble_k: int,
    ensemble_eps: Optional[float],
    *,
    env_name: str,
    member: str,
    name: str,
    q_bounds: Tuple[float, float],
    release_on_grid: bool = True,
) -> ExperimentRunnerConfig:
    """
    An ensemble of ensemble_k trainer oracles with a whole-table epsilon of ensemble_eps. The
    noise calibration (which depends on k and the number of training trajectories) is computed
    by the trainer oracle itself in train().

    member is "tabular" (QLearnerTrainerOracle on integer states) or "cql"
    (DeepDiscretizingTrainerOracle on feature vectors). release_on_grid makes "cql" members
    release Q values at env.features (the discretized release the DP noise is calibrated for);
    False queries the network at each exact observation, the undefended baseline, valid only
    with epsilon None and k = 1. Sepsis observations are already the release points.
    """
    if member not in ("tabular", "cql"):
        raise ValueError(f"Unknown ensemble member {member}")
    deep = member == "cql"
    gamma = GAMMA[env_name]
    env, data_oracle = make_env_and_data_oracle(env_name, discrete=not deep)

    # Builds a new, independent sub-oracle on each call
    def make_inner_trainer_oracle():
        if deep:
            return DeepDiscretizingTrainerOracle(DeepDiscretizingTrainerOracleConfig(
                device="cpu",
                env=env,
                data_oracle=data_oracle,
                points=env.features if release_on_grid else None,  # public points to release Q values for
                cql_alpha=1.0,
                target_bounds=(0, 1),  # returns are in [0, 1] in both environments
                buffer_size=200_000,
                learning_rate=3e-4,
                batch_size=128,
                target_updates=30,
                discount_factor=gamma,
                net_arch=[128, 128],
            ))
        return QLearnerTrainerOracle(
            QLearnerTrainerOracleConfig(
                env=env,
                discount_factor=gamma,
                data_oracle=data_oracle,
                q_learner=QLearner(
                    QLearnerConfig(
                        env=env,
                        alpha=0.001,
                        buffer_size=1_000_000_000,
                        discount_factor=gamma,
                        unseen_training_val=0,
                        unseen_inference_val=0,
                    )
                ),
            ),
        )

    trainer_oracle = EnsemblingTrainerOracle(
        EnsemblingTrainerOracleConfig(
            env=env,
            data_oracle=data_oracle,
            k=ensemble_k,
            trainer_factory=make_inner_trainer_oracle,
            q_bounds=q_bounds,
            epsilon=ensemble_eps,
            delta=1e-5,
        )
    )
    return _runner_config(name, env_name, env, trainer_oracle, EPOCHS[member])


# ---------- DP-APVI (Qiao & Wang 2022) ----------

GLUCOSE_REWARD_TABLE_PATH = "models/glucose_reward_table.npy"
DPAPVI_PENALTY_SCALE = 0.0  # the paper's Bernstein penalty (1.0) zeroes out Q on Sepsis


def dpapvi_reward_table(env_name: str, env: gym.Env):
    """
    DP-APVI's public reward model: None for Sepsis (it reads ICU-Sepsis's own reward table);
    for glucose, the mean reward of each grid transition in simulations independent of the
    training data (GlucoseDiscreteObs.public_reward_table), computed once and saved.
    """
    if env_name != "glucose":
        return None
    if not Path(GLUCOSE_REWARD_TABLE_PATH).exists():
        np.save(GLUCOSE_REWARD_TABLE_PATH, env.public_reward_table())
    return np.load(GLUCOSE_REWARD_TABLE_PATH)


def build_dpapvi_config(ensemble_k: int, ensemble_eps: Optional[float], *, env_name: str, name: str) -> ExperimentRunnerConfig:
    """
    DP-APVI: noisy counts of the member trajectories and pessimistic value iteration on the
    private model. ensemble_eps is the whole-release epsilon (None = exact counts, no privacy).
    ensemble_k is unused; it is accepted so run_sweep can call every builder the same way.
    """
    env, data_oracle = make_env_and_data_oracle(env_name, discrete=True)
    trainer_oracle = DPAVIVITrainerOracle(
        DPAVIVITrainerOracleConfig(
            env=env,
            # The attack reads it from here. DP-APVI is finite-horizon (T_max undiscounted rounds of
            # value iteration, values bounded by value_bound), so it handles the end of a day itself
            discount_factor=1,
            data_oracle=data_oracle,
            epsilon=ensemble_eps,
            T_max=T_MAX[env_name],
            value_bound=1.0,  # returns are in [0, 1] in both environments
            reward_table=dpapvi_reward_table(env_name, env),
            penalty_scale=DPAPVI_PENALTY_SCALE,
        )
    )
    return _runner_config(name, env_name, env, trainer_oracle, epochs=1)  # epochs unused: value iteration runs T_max rounds


# ---------- DP-SGD ----------

DPSGD_C = 1.0  # max norm of per-trajectory gradients


def build_dpsgd_config(ensemble_k: int, ensemble_eps: Optional[float], *, env_name: str, name: str) -> ExperimentRunnerConfig:
    """
    Offline CQL trained with DP-SGD at a whole-training epsilon of ensemble_eps (None = no
    noise or clipping). ensemble_k is unused, as for build_dpapvi_config.
    """
    env, data_oracle = make_env_and_data_oracle(env_name, discrete=False)
    trainer_oracle = DPSGDTrainerOracle(DPSGDTrainerOracleConfig(
        device="cpu",
        env=env,
        data_oracle=data_oracle,
        cql_alpha=1.0,
        target_bounds=(0, 1),  # returns are in [0, 1] in both environments
        learning_rate=3e-4,
        batch_size=128,
        target_updates=30,
        discount_factor=GAMMA[env_name],
        hidden_sizes=(128, 128),
        max_grad_norm=DPSGD_C,
        epsilon=ensemble_eps,
        trajectory_loss="sum",
    ))
    return _runner_config(name, env_name, env, trainer_oracle, EPOCHS["dpsgd"])
