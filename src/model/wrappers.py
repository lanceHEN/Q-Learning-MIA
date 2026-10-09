"""
Environment wrappers.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import solve_ivp


class TupleObsToBox(gym.ObservationWrapper):
    """
    Converts a Tuple(Discrete, ...) observation space to a flat Box by stacking
    the raw integer values into a numpy array. Unlike FlattenObservation, this
    does NOT one-hot encode — Discrete(32) contributes one float, not 32.

    Useful for making environments like Blackjack-v1 compatible with SB3's DQN,
    which requires a Box observation space.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Tuple), (
            "TupleObsToBox requires a Tuple observation space"
        )
        subspaces = env.observation_space.spaces
        lows  = np.array([s.start           for s in subspaces], dtype=np.float32)
        highs = np.array([s.start + s.n - 1 for s in subspaces], dtype=np.float32)
        self.observation_space = spaces.Box(low=lows, high=highs, dtype=np.float32)

    def observation(self, obs) -> np.ndarray:
        return np.array(obs, dtype=np.float32)

class ChemotherapyEnv(gym.Env):
    """
    Cancer chemotherapy dosing on a four-variable tumor-immune-normal-cell ODE:
    normal cells N, tumor cells T, immune cells I, drug concentration B (all
    normalized so a healthy patient has N near 1).

    Sources
      Model structure and all 16 parameter values: eqs. 37-40 and Table 8 of
      Padmanabhan, Meskin & Haddad, "Reinforcement learning-based control of drug
      dosing with applications to anesthesia and cancer therapy" (book chapter;
      the RL formulation was published in Math Biosci 2017). They take the model
      from De Pillis & Radunskaya (2001, 2003) and Batmani & Khaloozadeh (2013).
      Everything else follows AhnChemoEnv in the open-source DTR-Bench
      (Luo et al., 2024, arXiv:2405.18610; a preprint).

    From DTR-Bench, not from the chapter
      Saturating drug-kill term a*(1 - exp(-B))*x (the chapter uses a linear
      a*x*B with doses up to umax = 10); doses in [0, 1], evenly spaced; initial
      means (N, T, I, B) = (0.9, 0.2, 0.005, 0) with uniform +/-25% spread;
      6-hour step and 600 steps (150 days); reward N/N0 - T/T0 - dose computed on
      the state before the step; death at N < 0.7*N0 with reward -100.

    Ours
      The agent observes the full state plus the starting N0 and T0 (the reward
      and death rule depend on them). This makes the problem Markov but gives
      the agent more than a clinic could measure. The dynamics are deterministic
      and the ODE parameters fixed, so patients differ only in their start state.

    This is a standard stylized model of tumor-immune-drug dynamics, not a
    clinically validated patient simulator.
    """

    _P = dict(r2=1.0, b2=1.0, c4=1.0, a3=0.1,
              r1=1.5, b1=1.0, c2=0.5, c3=1.0, a2=0.3,
              s=0.33, rho=0.01, alpha=0.3, c1=1.0, d1=0.2, a1=0.2,
              d2=1.0)
    _INIT_MEAN = np.array([0.9, 0.2, 0.005, 0.0])
    _DT = 0.25
    _DEATH_FRACTION = 0.7
    _DEATH_REWARD = -100.0

    def __init__(self, max_steps: int = 600, init_spread: float = 0.25, n_actions: int = 5):
        super().__init__()
        self._max_steps = max_steps
        self._init_spread = init_spread
        self._doses = np.linspace(0.0, 1.0, n_actions)
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.array([2.0, 2.0, 5.0, 5.0, 2.0, 2.0], dtype=np.float32),
        )
        self._state = self._INIT_MEAN.copy()
        self._n0, self._t0 = self._INIT_MEAN[0], self._INIT_MEAN[1]
        self._steps = 0

    def _rhs(self, t, y, u):
        N, T, I, B = y
        p = self._P
        kill = 1.0 - np.exp(-B)
        return [
            p['r2'] * N * (1 - p['b2'] * N) - p['c4'] * T * N - p['a3'] * kill * N,
            p['r1'] * T * (1 - p['b1'] * T) - p['c2'] * I * T - p['c3'] * T * N - p['a2'] * kill * T,
            p['s'] + p['rho'] * I * T / (p['alpha'] + T) - p['c1'] * I * T - p['d1'] * I - p['a1'] * kill * I,
            -p['d2'] * B + u,
        ]

    def _obs(self) -> np.ndarray:
        return np.concatenate([self._state, [self._n0, self._t0]]).astype(np.float32)

    def reset(self, **kwargs):
        jitter = 1.0 + np.random.uniform(-self._init_spread, self._init_spread, size=4)
        self._state = self._INIT_MEAN * jitter
        self._state[3] = 0.0
        self._n0, self._t0 = self._state[0], self._state[1]
        self._steps = 0
        return self._obs(), {}

    def step(self, action: int):
        dose = float(self._doses[int(action)])
        reward = self._state[0] / self._n0 - self._state[1] / self._t0 - dose
        sol = solve_ivp(self._rhs, (0.0, self._DT), self._state, args=(dose,))
        self._state = np.maximum(sol.y[:, -1], 0.0)
        self._steps += 1
        terminated = bool(self._state[0] < self._DEATH_FRACTION * self._n0)
        if terminated:
            reward = self._DEATH_REWARD
        truncated = (not terminated) and self._steps >= self._max_steps
        return self._obs(), float(reward), terminated, truncated, {}


class HIVTreatmentEnv(gym.Env):
    """
    HIV structured-treatment-interruption (STI) control on the six-variable ODE
    model of Adams, Banks, Kwon & Tran (2004), with the RL task formulated as in
    Ernst, Stan, Goncalves & Wehenkel (2006).

    Sources
      Model (eq. 1) and all parameter values (Table 1): Adams et al., Math Biosci
      Eng 1(2):223-241, including the RTI efficacy reduction f = 0.34 for the
      macrophage population and the virus-consumption term in dV/dt.
      Task: Ernst et al., "Clinical data based optimal STI strategies for HIV: a
      reinforcement learning approach" (CDC 2006): four actions (RTI and PI each
      on/off, with efficacy eps1 = 0.7 and eps2 = 0.3 when on), a decision every
      5 days, 1000 days per patient (200 steps), and patients starting at the
      "non-healthy" equilibrium (T1, T2, T1*, T2*, V, E) =
      (163573, 5, 11945, 46, 63919, 24). Their instantaneous cost is
      c = Q*V + R1*eps1^2 + R2*eps2^2 - S*E with Q = 0.1, R1 = 20000,
      R2 = 2000, S = 1000, and the reward here is -c. (Adams et al. use
      R2 = 20000 in their own simulations; Ernst et al. use 2000.) As in Ernst et
      al., the cost is evaluated on the state at decision time. The dynamics are
      deterministic.

    Ours, not from the sources
      `patient_noise` (default 0.1): each of the six starting values is
      multiplied by lognormal noise with this sigma, so patients start near the
      equilibrium instead of exactly on it. Both papers start every patient
      exactly at the equilibrium; set it to 0 for that. The observation is log10
      of the six variables so a neural network sees a narrow range.
      `reward_scale` multiplies the reward by a positive constant (1e-4), which
      leaves the optimal policy unchanged but keeps magnitudes workable for DQN.
      Episodes end by time limit (truncated), as in the 1000-day protocol.
    """

    # Table 1 of Adams et al. (2004)
    _P = dict(
        lam1=10000., lam2=31.98,
        d1=0.01,     d2=0.01,
        k1=8e-7,     k2=1e-4,
        f=0.34,
        delta=0.7,
        m1=1e-5,     m2=1e-5,
        NT=100.,     c=13.,
        rho1=1.,     rho2=1.,
        lam_E=1.,    b_E=0.3,
        K_b=100.,    d_E=0.25,
        K_d=500.,    delta_E=0.1,
    )

    # (eps1 = RTI efficacy, eps2 = PI efficacy) per action index
    _DRUGS = [(0.0, 0.0), (0.7, 0.0), (0.0, 0.3), (0.7, 0.3)]

    # Ernst et al. (2006) cost weights
    _Q, _R1, _R2, _S = 0.1, 20000., 2000., 1000.

    # "Non-healthy" equilibrium of Adams et al. (2004) and Ernst et al. (2006)
    _STEADY = np.array([163573., 5., 11945., 46., 63919., 24.])

    def __init__(self, episode_length: int = 200, days_per_step: int = 5,
                 patient_noise: float = 0.1, reward_scale: float = 1e-4):
        super().__init__()
        self._episode_length = episode_length
        self._days_per_step = days_per_step
        self._patient_noise = patient_noise
        self._reward_scale = reward_scale

        self.action_space = spaces.Discrete(4)
        # Observations are log10(state); bounds cover ~1e-8 to 1e7
        self.observation_space = spaces.Box(
            low=np.full(6, -8.0, dtype=np.float32),
            high=np.full(6,  7.0, dtype=np.float32),
        )
        self._state: np.ndarray = self._STEADY.copy()
        self._step_count: int = 0

    def _ode(self, t, y, eps1, eps2):
        T1, T2, T1s, T2s, V, E = y
        p = self._P
        inf = T1s + T2s
        rtI1 = (1 - eps1)
        rtI2 = (1 - p['f'] * eps1)
        return [
            p['lam1'] - p['d1'] * T1 - rtI1 * p['k1'] * V * T1,
            p['lam2'] - p['d2'] * T2 - rtI2 * p['k2'] * V * T2,
            rtI1 * p['k1'] * V * T1 - p['delta'] * T1s - p['m1'] * E * T1s,
            rtI2 * p['k2'] * V * T2 - p['delta'] * T2s - p['m2'] * E * T2s,
            ((1 - eps2) * p['NT'] * p['delta'] * inf - p['c'] * V
             - (rtI1 * p['rho1'] * p['k1'] * T1 + rtI2 * p['rho2'] * p['k2'] * T2) * V),
            (p['lam_E']
             + p['b_E'] * inf / (inf + p['K_b']) * E
             - p['d_E'] * inf / (inf + p['K_d']) * E
             - p['delta_E'] * E),
        ]

    def _obs(self) -> np.ndarray:
        return np.log10(np.maximum(self._state, 1e-8)).astype(np.float32)

    def reset(self, **kwargs):
        if self._patient_noise > 0:
            noise = np.random.lognormal(0.0, self._patient_noise, size=6)
            self._state = np.maximum(self._STEADY * noise, 1e-8)
        else:
            self._state = self._STEADY.copy()
        self._step_count = 0
        return self._obs(), {}

    def step(self, action: int):
        eps1, eps2 = self._DRUGS[int(action)]
        V, E = self._state[4], self._state[5]
        cost = self._Q * V + self._R1 * eps1 ** 2 + self._R2 * eps2 ** 2 - self._S * E
        reward = float(-cost * self._reward_scale)

        sol = solve_ivp(
            self._ode,
            [0, self._days_per_step],
            self._state,
            args=(eps1, eps2),
            method='LSODA',
            rtol=1e-8,
            atol=1e-8,
        )
        self._state = np.maximum(sol.y[:, -1], 1e-8)
        self._step_count += 1

        truncated = self._step_count >= self._episode_length
        return self._obs(), reward, False, truncated, {}
