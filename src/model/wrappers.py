"""
Environment wrappers.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import solve_ivp

class SepsisStateVectorObs(gym.ObservationWrapper):
    """
    Replaces ICU-Sepsis's integer state with a feature vector: the state's cluster center
    (47 normalized patient features from the AI Clinician pipeline, in the order of
    icu_sepsis_helpers.mdp_creation.create_rl_table: 4 binary indicators minus 0.5, 32
    z-scored, 11 z-scored after log(x + 0.1)), followed by one indicator per terminal
    state. The indicators are needed because the terminal states' cluster centers are all
    zero, which would make death, survival, and an average patient indistinguishable.

    encode() gives the same vector as a hashable tuple, for trajectories generated on the
    integer environment (see FeatureMappedDataOracle). is_terminal_obs() lets the deep
    trainer oracles recognize terminal states from their vectors.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        from icu_sepsis.utils.constants import STATES_TERMINAL
        unwrapped = env.unwrapped
        centers = np.asarray(unwrapped.state_cluster_centers, dtype=np.float32) # 716 * 47
        self.terminal_states = sorted(STATES_TERMINAL) # integers
        indicators = np.zeros((len(centers), len(self.terminal_states)), dtype=np.float32) # 716 * 3
        for i, state in enumerate(self.terminal_states):
            indicators[state, i] = 1.0
        self.features = np.hstack([centers, indicators]) # 716 * 50
        self.n_center_features = centers.shape[1]
        self.observation_space = spaces.Box(
            low=self.features.min(axis=0), high=self.features.max(axis=0), dtype=np.float32)

    def observation(self, obs) -> np.ndarray:
        return self.features[int(obs)]

    def encode(self, state: int) -> tuple:
        return tuple(float(x) for x in self.features[int(state)])

    def is_terminal_obs(self, obs) -> bool:
        return bool(np.asarray(obs, dtype=np.float32).flatten()[self.n_center_features:].max() > 0.5)

class _FastParams:
    """
    simglucose's patient parameters as plain attributes. The patient ODE reads them as
    attributes of a pandas Series about 500,000 times per simulated day, which takes most
    of the simulation time; plain attributes give identical trajectories ~7x faster.
    iloc is kept for T1DPatient.reset, which reads the initial state with it.
    """

    def __init__(self, series):
        self.__dict__.update(series.to_dict())
        self.iloc = series.iloc


try:
    from numba import njit as _njit
except ImportError:  # runs the same code as plain Python, much more slowly
    def _njit(*args, **kwargs):
        return lambda f: f


# simglucose's patient parameters used by the ODE, packed into an array in this order
_PATIENT_PARAMS = ("kmax", "kmin", "kabs", "b", "d", "f", "BW", "kp1", "kp2", "kp3", "Fsnc", "ke1",
                   "ke2", "k1", "k2", "Vm0", "Vmx", "Km0", "m1", "m2", "m4", "m30", "ka1", "ka2",
                   "kd", "Vi", "p2u", "Ib", "ki", "ksc", "Vg")
(_KMAX, _KMIN, _KABS, _B, _D, _F, _BW, _KP1, _KP2, _KP3, _FSNC, _KE1, _KE2, _K1, _K2, _VM0, _VMX,
 _KM0, _M1, _M2, _M4, _M30, _KA1, _KA2, _KD, _VI, _P2U, _IB, _KI, _KSC, _VG) = range(len(_PATIENT_PARAMS))
_EAT_RATE = 5.0  # g/min, T1DPatient.EAT_RATE


@_njit()
def _patient_rhs(x, d, insulin, P, Dbar, out):
    """simglucose's T1DPatient.model (Dalla Man et al.), unchanged; P from _PATIENT_PARAMS."""
    qsto = x[0] + x[1]
    out[0] = -P[_KMAX] * x[0] + d
    if Dbar > 0:
        aa = 5 / (2 * Dbar * (1 - P[_B]))
        cc = 5 / (2 * Dbar * P[_D])
        kgut = P[_KMIN] + (P[_KMAX] - P[_KMIN]) / 2 * (
            np.tanh(aa * (qsto - P[_B] * Dbar)) - np.tanh(cc * (qsto - P[_D] * Dbar)) + 2)
    else:
        kgut = P[_KMAX]
    out[1] = P[_KMAX] * x[0] - x[1] * kgut
    out[2] = kgut * x[1] - P[_KABS] * x[2]
    rat = P[_F] * P[_KABS] * x[2] / P[_BW]
    egpt = P[_KP1] - P[_KP2] * x[3] - P[_KP3] * x[8]
    et = P[_KE1] * (x[3] - P[_KE2]) if x[3] > P[_KE2] else 0.0
    out[3] = max(egpt, 0.0) + rat - P[_FSNC] - et - P[_K1] * x[3] + P[_K2] * x[4]
    uidt = (P[_VM0] + P[_VMX] * x[6]) * x[4] / (P[_KM0] + x[4])
    out[4] = -uidt + P[_K1] * x[3] - P[_K2] * x[4]
    out[5] = -(P[_M2] + P[_M4]) * x[5] + P[_M1] * x[9] + P[_KA1] * x[10] + P[_KA2] * x[11]
    it = x[5] / P[_VI]
    out[6] = -P[_P2U] * x[6] + P[_P2U] * (it - P[_IB])
    out[7] = -P[_KI] * (x[7] - it)
    out[8] = -P[_KI] * (x[8] - x[7])
    out[9] = -(P[_M1] + P[_M30]) * x[9] + P[_M2] * x[5]
    out[10] = insulin - (P[_KA1] + P[_KD]) * x[10]
    out[11] = P[_KD] * x[10] - P[_KA2] * x[11]
    out[12] = -P[_KSC] * x[12] + P[_KSC] * x[3]
    for i in (3, 4, 5, 9, 10, 11, 12):  # these compartments cannot go negative
        if x[i] < 0:
            out[i] = 0.0


@_njit()
def _simulate_patient(x, eat, P, insulin, meals, check_range, bg_out):
    """
    Advances the patient one minute per entry of `meals` (g announced that minute), as
    T1DPatient.step does: announced meals are eaten at _EAT_RATE, the stomach contents at
    the start of eating are tracked for gut absorption, and the ODE is advanced with one
    fixed RK4 step per minute (simglucose uses scipy's adaptive dopri5; blood glucose agrees
    to within 0.13 mg/dL). insulin is constant (pmol/kg/min). x and eat (last_Qsto,
    last_foodtaken in g, eating flag, planned meal in g, last minute's CHO in g) are updated
    in place; bg_out gets blood glucose (Gsub, mg/dL) after each minute. Returns the number
    of minutes simulated, stopping early if check_range and glucose leaves [10, 600].
    """
    k1 = np.empty(13); k2 = np.empty(13); k3 = np.empty(13); k4 = np.empty(13); tmp = np.empty(13)
    last_qsto, last_food, eating, planned, last_cho = eat[0], eat[1], eat[2] > 0, eat[3], eat[4]
    n = 0
    for i in range(meals.shape[0]):
        planned += meals[i]
        if planned > 0:
            cho = min(_EAT_RATE, planned)
            planned = max(0.0, planned - cho)
        else:
            cho = 0.0
        if cho > 0 and last_cho <= 0:
            last_qsto = x[0] + x[1]
            last_food = 0.0
            eating = True
        if eating:
            last_food += cho
        if cho <= 0 and last_cho > 0:
            eating = False
        last_cho = cho
        d = cho * 1000.0
        dbar = last_qsto + last_food * 1000.0
        _patient_rhs(x, d, insulin, P, dbar, k1)
        for j in range(13):
            tmp[j] = x[j] + 0.5 * k1[j]
        _patient_rhs(tmp, d, insulin, P, dbar, k2)
        for j in range(13):
            tmp[j] = x[j] + 0.5 * k2[j]
        _patient_rhs(tmp, d, insulin, P, dbar, k3)
        for j in range(13):
            tmp[j] = x[j] + k3[j]
        _patient_rhs(tmp, d, insulin, P, dbar, k4)
        for j in range(13):
            x[j] += (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6.0
        bg = x[12] / P[_VG]
        bg_out[i] = bg
        n = i + 1
        if check_range and (bg < 10 or bg > 600):
            break
    eat[0], eat[1], eat[2], eat[3], eat[4] = last_qsto, last_food, 1.0 if eating else 0.0, planned, last_cho
    return n


class GlucoseEnv(gym.Env):
    """
    Basal insulin control for type 1 diabetes on simglucose (Jinyu Xie, 2018), an
    implementation of the UVA/Padova 2008 simulator (Dalla Man et al.), changed only where
    the MIA pipeline needs it.

    As in simglucose
      Patient model and virtual patients, Dexcom CGM noise, Insulet pump, random meal
      scenario (up to 6 meals a day at random times and sizes, not observed by the agent),
      random start hour, random initial glucose, basal-only control (no meal boluses, as in
      simglucose's gym environment), and termination when blood glucose leaves the
      simulator's valid range (< 10 or > 600 mg/dL).

    Changed for the pipeline
      - Speed: the patient ODE is advanced with one fixed RK4 step per minute, compiled with
        numba (_simulate_patient), instead of scipy's adaptive dopri5 (~30x faster). The
        equations, meal and eating logic, initial states, sensor noise and meals are
        simglucose's; blood glucose agrees with simglucose's to within 0.13 mg/dL (46 one-day
        episodes from identical seeds, glucose 23-382 mg/dL), well below the CGM noise.
      - Seeds are drawn from numpy's global generator, so generate_trajectories(seed) is
        reproducible and episodes do not repeat (simglucose's gym environment draws them
        from only 1000 values).
      - Discrete actions: `n_actions` basal rates, multiples (0 to `max_basal_multiple`)
        of the patient's steady-state basal, each held for `decision_minutes`; episodes are
        truncated after `max_steps` decisions (default 48 x 30 min = one day).
      - Bounded reward: one minus simglucose's risk index (Kovatchev's blood glucose risk
        index, 0-100 over the CGM range 39-600 mg/dL), averaged over the step's CGM
        readings, divided by 100 * max_steps, so returns are in [0, 1].
      - Multi-dimensional observation, from simglucose's own outputs (scaled to be of order
        1, since release points are matched by L2 distance):
          0  CGM / 100 mg/dL
          1  CGM change over the last 30 min / 20 mg/dL
          2  insulin delivered over the last 30 min, in multiples of the steady-state basal
          3  insulin delivered over the 90 min before that, same units
        A pre-roll of `preroll_minutes` on the steady-state basal fills this history.
        `features` is a public grid of these observations to release Q values for, and
        grid_index() gives an observation's nearest grid point; every method uses this one
        grid (see grid_points and GlucoseDiscreteObs).
      - Patients are drawn uniformly from `patients` (simglucose's gym default is one
        patient, adolescent#001; it also accepts a list).
    """

    ADULTS = tuple(f"adult#{i:03d}" for i in range(1, 11))
    _CGM_STEP = 3  # minutes between CGM readings (Dexcom)
    _SCALES = np.array([100.0, 20.0, 1.0, 1.0], dtype=np.float32)
    GRID_CGM = (70, 110, 140, 170, 200, 240, 290, 350)  # mg/dL
    GRID_TREND = (-30, -10, 10, 30)  # mg/dL per 30 min
    _tables = None  # simglucose's parameter tables, read once per process

    def __init__(self, patients=ADULTS, decision_minutes: int = 30, max_steps: int = 48,
                 n_actions: int = 5, max_basal_multiple: float = 2.0, preroll_minutes: int = 120):
        super().__init__()
        self._patients = list(patients)
        self._decision_minutes = decision_minutes
        self._max_steps = max_steps
        # Episodes reaching time_limit steps are truncated, not terminated: learners bootstrap
        # through the last step (see trainer_oracle.ends_episode)
        self.time_limit = max_steps
        self._preroll = preroll_minutes
        self.basal_multiples = np.linspace(0.0, max_basal_multiple, n_actions)
        self.action_space = spaces.Discrete(n_actions)
        # Bounds from simglucose's parameters: CGM readings are clipped to the sensor's range,
        # so a 30-min change is at most its width; insulin is at most the largest basal
        # multiple (or the pre-roll's 1x), plus the pump's rounding of basal rates to its
        # increment (at most half an increment, relative to the smallest patient's basal)
        tables = self._load_tables()
        sensor = tables["sensor"].loc[tables["sensor"].Name == "Dexcom"].squeeze()
        pump = tables["pump"].loc[tables["pump"].Name == "Insulet"].squeeze()
        patient_rows = tables["patients"][tables["patients"].Name.isin(self._patients)]
        min_basal = float((patient_rows.u2ss * patient_rows.BW / 6000).min())  # U/min
        rounding = pump.inc_basal / 2 / 6000 / min_basal  # pmol/min -> U/min (U2PMOL = 6000), as a multiple
        max_insulin = max(max_basal_multiple, 1.0) + rounding
        width = sensor["max"] - sensor["min"]
        low = np.array([sensor["min"], -width, 0.0, 0.0]) / self._SCALES
        high = np.array([sensor["max"], width, max_insulin, max_insulin]) / self._SCALES
        self.observation_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32))
        self._grid_centers = [
            np.array(self.GRID_CGM) / self._SCALES[0],
            np.array(self.GRID_TREND) / self._SCALES[1],
            self.basal_multiples / self._SCALES[2],
            self.basal_multiples / self._SCALES[3],
        ]
        self._grid_shape = tuple(len(c) for c in self._grid_centers)
        self.features = self.grid_points()

    def grid_points(self) -> np.ndarray:
        """
        The public grid of observations that every method releases Q values for: the deep
        methods' points (DeepDiscretizingTrainerOracle) and the tabular methods' states
        (GlucoseDiscreteObs), so all methods answer on the same states. CGM (8 values,
        covering the behavior policy's 1st-99th percentiles of 95-323 mg/dL for adult
        patients) x 30-min change (4) x insulin over the last 30 min (the basal multiples,
        which it takes exactly) x insulin over the 90 min before (the basal multiples; it is a
        mean of three, so it is snapped to the nearest): 800 points with the default 5
        actions, about the size of ICU-Sepsis's 716, so DP-APVI's S x A x S count arrays stay
        small. Row i is the point of state i (C order, as grid_index).
        """
        grids = np.meshgrid(*self._grid_centers, indexing="ij")
        return np.stack([g.ravel() for g in grids], axis=1).astype(np.float32)

    def grid_index(self, obs) -> int:
        """
        The index of the grid point nearest to an observation (by L2 distance, as
        DeepDiscretizingTrainerOracle matches points). The grid is a product, so the nearest
        point is found one feature at a time.
        """
        values = np.asarray(obs, dtype=float).ravel()
        idx = tuple(int(np.abs(c - v).argmin()) for c, v in zip(self._grid_centers, values))
        return int(np.ravel_multi_index(idx, self._grid_shape))

    @classmethod
    def _load_tables(cls):
        if cls._tables is None:
            import pandas as pd
            import pkg_resources
            path = lambda f: pkg_resources.resource_filename("simglucose", f"params/{f}")
            cls._tables = {
                "patients": pd.read_csv(path("vpatient_params.csv")),
                "sensor": pd.read_csv(path("sensor_params.csv")),
                "pump": pd.read_csv(path("pump_params.csv")),
            }
        return cls._tables

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        from datetime import datetime, timedelta
        from simglucose.patient.t1dpatient import T1DPatient
        from simglucose.sensor.cgm import CGMSensor
        from simglucose.simulation.scenario_gen import RandomScenario

        tables = self._load_tables()
        name = self._patients[np.random.randint(len(self._patients))]
        patient_seed, sensor_seed, meal_seed = (int(s) for s in np.random.randint(0, 2**31 - 1, size=3))
        hour = np.random.randint(0, 24)  # simglucose's random start hour
        row = tables["patients"].loc[tables["patients"].Name == name].squeeze()
        params = _FastParams(row)
        self._basal = float(row.u2ss * row.BW / 6000)  # steady-state basal, U/min
        self._P = np.array([getattr(params, n) for n in _PATIENT_PARAMS], dtype=np.float64)
        # simglucose's initial state (random initial glucose); the ODE is advanced by _simulate_patient
        self._x = T1DPatient(params, random_init_bg=True, seed=patient_seed).init_state.astype(np.float64)
        self._eat = np.array([self._x[0] + self._x[1], 0.0, 0.0, 0.0, 0.0])
        sensor_row = tables["sensor"].loc[tables["sensor"].Name == "Dexcom"].squeeze()
        self._noise = CGMSensor(sensor_row, seed=sensor_seed)._noise_generator
        self._cgm_range = (float(sensor_row["min"]), float(sensor_row["max"]))
        next(self._noise)  # simglucose's environment takes a reading at time 0
        pump_row = tables["pump"].loc[tables["pump"].Name == "Insulet"].squeeze()
        self._pump = (float(pump_row.inc_basal), float(pump_row.min_basal), float(pump_row.max_basal))
        start = datetime(2018, 1, 1, hour) - timedelta(minutes=self._preroll)
        self._scenario = RandomScenario(start_time=start, seed=meal_seed)
        self._clock0 = start.hour * 60 + start.minute  # minute of the day at time 0

        self._minute = 0
        self._insulin = []  # units delivered in each minute
        self._cgm = []  # one reading every _CGM_STEP minutes
        self._steps = 0
        self._run(self._basal, self._preroll, check_range=False)
        return self._obs(), {}

    def _meals(self, minutes: int) -> np.ndarray:
        """
        Meals announced (g) in each of the next `minutes` minutes, as RandomScenario.get_action
        gives them, including drawing a new day's meals at midnight.
        """
        clocks = (self._clock0 + self._minute + np.arange(minutes)) % 1440
        meals = np.zeros(minutes)
        start = 0
        for stop in list(np.flatnonzero(clocks == 0)) + [minutes]:
            day = self._scenario.scenario["meal"]
            for t, grams in zip(day["time"], day["amount"]):
                meals[start:stop][clocks[start:stop] == t] = grams
            if stop < minutes:
                self._scenario.scenario = self._scenario.create_scenario()
            start = stop
        return meals

    def _run(self, basal: float, minutes: int, check_range: bool) -> bool:
        """
        Simulates `minutes` minutes at a basal rate (U/min), recording insulin and CGM readings
        as simglucose's environment does; returns whether blood glucose left [10, 600] mg/dL.
        """
        inc, lo, hi = self._pump  # the pump rounds basal rates to its increment (InsulinPump.basal)
        rate = min(max(np.round(basal * 6000 / inc) * inc / 6000, lo), hi)
        bg = np.empty(minutes)
        n = _simulate_patient(self._x, self._eat, self._P, rate * 6000 / self._P[_BW],
                              self._meals(minutes), check_range, bg)
        self._insulin.extend([rate] * n)
        cgm_lo, cgm_hi = self._cgm_range
        for i in range(n):
            self._minute += 1
            if self._minute % self._CGM_STEP == 0:
                self._cgm.append(min(max(bg[i] + next(self._noise), cgm_lo), cgm_hi))
        return check_range and n > 0 and not (10 <= bg[n - 1] <= 600)

    def _obs(self) -> np.ndarray:
        cgm, d = self._cgm, self._decision_minutes
        per_basal_minute = lambda units, minutes: units / (self._basal * minutes)
        raw = np.array([
            cgm[-1],
            cgm[-1] - cgm[-1 - 30 // self._CGM_STEP],
            per_basal_minute(sum(self._insulin[-d:]), d),
            per_basal_minute(sum(self._insulin[-4 * d:-d]), 3 * d),
        ])
        return (raw / self._SCALES).astype(np.float32)

    def step(self, action: int):
        basal = float(self.basal_multiples[int(action)]) * self._basal
        n_before = len(self._cgm)
        out_of_range = self._run(basal, self._decision_minutes, check_range=True)
        readings = np.array(self._cgm[n_before:])
        if len(readings):
            # simglucose's risk index (risk_index) of each reading: 10 * f(BG)^2
            f_bg = 1.509 * (np.log(readings) ** 1.084 - 5.381)
            risk = float(np.mean(10 * f_bg ** 2))
        else:
            risk = 100.0
        reward = max(0.0, 1.0 - risk / 100.0) / self._max_steps
        self._steps += 1
        truncated = (not out_of_range) and self._steps >= self._max_steps
        return self._obs(), float(reward), out_of_range, truncated, {}


class GlucoseDiscreteObs(gym.ObservationWrapper):
    """
    GlucoseEnv with a discrete state, for tabular learners (QLearnerTrainerOracle, DP-APVI):
    the index of the nearest point of GlucoseEnv's grid (see GlucoseEnv.grid_points), the
    same points the deep methods release Q values for.

    When glucose leaves the valid range the episode terminates and the state is end_state,
    one past the grid points. Like ICU-Sepsis's terminal states it is never acted in, so
    tabular learners value it 0. The end of a day is a truncation (GlucoseEnv.time_limit),
    not a termination: the state is the ordinary grid state, so learners bootstrap through
    it, with a discount below 1 keeping values bounded. Ending each day in end_state instead
    (end_on_truncation) makes the end of an episode depend on a clock the state does not
    contain, and tabular values then grow past any achievable return; it is kept only to
    reproduce the saved glucose data oracle, which was trained that way.

    discretize() maps a GlucoseEnv observation to its state, for data generated on the
    continuous environment (see FeatureMappedDataOracle); features holds each state's
    grid point, in GlucoseEnv's scaled units.
    """

    def __init__(self, env: gym.Env, end_on_truncation: bool = False):
        super().__init__(env)
        self._base = env.unwrapped
        self._end_on_truncation = end_on_truncation
        self.features = self._base.features
        self.end_state = len(self.features)
        self.observation_space = spaces.Discrete(len(self.features) + 1)

    def discretize(self, obs) -> int:
        return self._base.grid_index(obs)

    def observation(self, obs) -> int:
        return self.discretize(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ends = terminated or (truncated and self._end_on_truncation)
        state = self.end_state if ends else self.discretize(obs)
        return state, reward, terminated, truncated, info

    def public_reward_table(self, n_episodes: int = 2000, seed: int = 0) -> np.ndarray:
        """
        A public model of the reward of each transition between states (S x S, in the
        environment's reward units), for algorithms that plan with a known reward (DP-APVI):
        the mean reward of each (s, s') over n_episodes simulated with uniformly random actions
        on a copy of this environment, with its own seed, so it uses no training data. Rewards
        depend on glucose, which s and s' record to the nearest grid value, so the action adds
        little once s' is known. A transition never simulated gets the reward at the midpoint of
        the two states' grid glucose (the start state's for end_state), simglucose's risk index
        as in GlucoseEnv.step. Leaves numpy's global random state unchanged.
        """
        import copy
        env = copy.deepcopy(self)
        n = self.observation_space.n
        sums, counts = np.zeros((n, n)), np.zeros((n, n))
        rng_state = np.random.get_state()
        np.random.seed(seed)
        try:
            for _ in range(n_episodes):
                s, _ = env.reset()
                while True:
                    s2, r, terminated, truncated, _ = env.step(np.random.randint(env.action_space.n))
                    sums[s, s2] += r
                    counts[s, s2] += 1
                    s = s2
                    if terminated or truncated:
                        break
        finally:
            np.random.set_state(rng_state)

        cgm = self.features[:, 0] * self._base._SCALES[0]
        start = np.append(cgm, np.nan)[:, None]  # end_state has no glucose of its own
        end = np.append(cgm, np.nan)[None, :]
        glucose = np.where(np.isnan(end), start, (start + end) / 2)
        f_bg = 1.509 * (np.log(np.nan_to_num(glucose, nan=112.5)) ** 1.084 - 5.381)
        formula = np.maximum(0.0, 1.0 - 10 * f_bg ** 2 / 100.0) / self._base._max_steps
        return np.where(counts > 0, sums / np.maximum(counts, 1), formula)
