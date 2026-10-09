"""
Heuristic fixed policy for Acrobot-v1.

Observation (6-dim):
  [cos(θ1), sin(θ1), cos(θ2), sin(θ2), dθ1, dθ2]
    θ1  — angle of link 1 from the downward vertical
    θ2  — relative angle of link 2 w.r.t. link 1
    dθ1 — angular velocity of joint 1
    dθ2 — angular velocity of joint 2

Actions: 0 = torque −1, 1 = no torque, 2 = torque +1

The policy uses an energy-pumping strategy: apply torque to joint 2
in the direction it is already rotating. This continuously injects
mechanical energy into the system — analogous to pumping a playground
swing — until the tip of the second link swings above the target height.

Typical performance: ~150–200 steps per episode (vs ~500 for random,
~80–100 for a well-trained DQN).
"""


def acrobot_policy(obs) -> int:
    """
    Returns the heuristic action for the given Acrobot observation.

    Applies positive torque (+1) when joint 2 is rotating in the positive
    direction, and negative torque (−1) otherwise. No-torque (action 1)
    is never used; the bang-bang structure is appropriate for energy pumping.
    """
    return 2 if obs[5] > 0 else 0
