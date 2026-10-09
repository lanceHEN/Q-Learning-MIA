"""
Optimal safe-path policy for the CliffWalking-v0 environment.

Grid: 4 rows × 12 columns (48 states), state = row * 12 + col.
Start: (3, 0) = state 36.  Goal: (3, 11) = state 47.
Cliff: row 3, col 1–10 (states 37–46) — stepping on one gives −100 and resets.

Actions: 0=Up, 1=Right, 2=Down, 3=Left.

Safe optimal path (reward −13):
  (3,0) → Up → (2,0) → Right ×11 → (2,11) → Down → (3,11)

For any off-path state the policy navigates back to row 2 first, then
proceeds right to col 11, then descends to the goal.
"""

NCOLS = 12


def _decode(state: int):
    return divmod(state, NCOLS)  # (row, col)


def cliffwalking_policy(state: int) -> int:
    """
    Returns the optimal safe-path action for the given CliffWalking-v0 state.

    Args:
        state: Integer state in [0, 47].

    Returns:
        Action in {0=Up, 1=Right, 2=Down, 3=Left}.
    """
    row, col = _decode(state)

    # On the bottom row: go up to leave the cliff zone
    if row == 3:
        return 0  # Up

    # On an upper row and not yet at the rightmost column: go right
    if col < 11:
        return 1  # Right

    # At col 11 and above the goal row: go down
    return 2  # Down
