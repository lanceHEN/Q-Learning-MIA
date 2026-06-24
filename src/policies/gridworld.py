"""
Simple deterministic policy for the GridWorld environment that moves
directly toward the bottom-right goal.
"""
from typing import Callable


def make_gridworld_policy(size: int) -> Callable[[int], int]:
    """Returns a policy function for a GridWorld of the given size."""
    goal_row, goal_col = size - 1, size - 1

    def policy(obs: int) -> int:
        row, col = obs // size, obs % size
        if row < goal_row:
            return 1  # down
        if col < goal_col:
            return 3  # right
        if row > goal_row:
            return 0  # up
        return 2      # left

    return policy
