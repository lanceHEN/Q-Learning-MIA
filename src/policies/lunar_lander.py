"""
Heuristic policy for LunarLander-v2/v3.

Observation (8-dim):
  [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]

Actions: 0=do nothing, 1=left engine, 2=main engine, 3=right engine

This is the standard OpenAI PD-controller heuristic. It achieves ~200 mean
reward (the published "solved" threshold) and is suitable as a fixed data
oracle policy for offline CQL experiments.
"""
import numpy as np


def lunar_lander_policy(obs) -> int:
    """
    Returns the heuristic action for the given LunarLander observation.

    The policy uses two PD controllers:
      - angle controller: steers the lander to face toward the landing pad
      - hover controller: targets a descent height proportional to horizontal
        offset (so the lander descends to land when centred)
    """
    # Target angle: point toward centre, proportional to x position and vx
    angle_targ = obs[0] * 0.5 + obs[2] * 1.0
    angle_targ = np.clip(angle_targ, -0.4, 0.4)   # cap at ±0.4 rad (~22°)

    # Target height: descend proportionally to horizontal distance from pad
    hover_targ = 0.55 * abs(obs[0])

    # PD controllers
    angle_todo = (angle_targ - obs[4]) * 0.5 - obs[5] * 1.0
    hover_todo  = (hover_targ - obs[1]) * 0.5 - obs[3] * 0.5

    # Smooth landing once legs touch
    if obs[6] or obs[7]:
        angle_todo = 0
        hover_todo = -obs[3] * 0.5

    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        return 2   # main engine
    elif angle_todo < -0.05:
        return 3   # right engine
    elif angle_todo > 0.05:
        return 1   # left engine
    return 0       # do nothing
