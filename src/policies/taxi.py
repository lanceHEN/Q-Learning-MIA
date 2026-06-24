"""
Optimal heuristic policy for the Taxi-v3 environment, using BFS over the
grid (respecting walls) to navigate to the passenger then the destination.
"""
import random
from collections import deque

LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B

WALLS = {
    # (row, col, direction): East=2, West=3
    (0, 1, 2), (0, 2, 3),
    (1, 1, 2), (1, 2, 3),
    (3, 0, 2), (3, 1, 3),
    (4, 0, 2), (4, 1, 3),
    (3, 2, 2), (3, 3, 3),
    (4, 2, 2), (4, 3, 3),
}

MOVES = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}  # S, N, E, W


def decode_state(state: int):
    dest = state % 4;        state //= 4
    pass_loc = state % 5;    state //= 5
    taxi_col = state % 5;    state //= 5
    taxi_row = state
    return taxi_row, taxi_col, pass_loc, dest


def _bfs(start, target):
    if start == target:
        return None
    queue = deque([(start, [])])
    visited = {}
    optimal_first_actions = []
    optimal_length = None
    while queue:
        (row, col), path = queue.popleft()
        if optimal_length is not None and len(path) > optimal_length:
            break
        for action, (dr, dc) in MOVES.items():
            if (row, col, action) in WALLS:
                continue
            nr, nc = row + dr, col + dc
            if not (0 <= nr < 5 and 0 <= nc < 5):
                continue
            new_path = path + [action]
            if (nr, nc) == target:
                optimal_length = len(new_path)
                if new_path[0] not in optimal_first_actions:
                    optimal_first_actions.append(new_path[0])
                continue
            dist = len(new_path)
            if (nr, nc) not in visited or visited[(nr, nc)] >= dist:
                visited[(nr, nc)] = dist
                queue.append(((nr, nc), new_path))
    return random.choice(optimal_first_actions) if optimal_first_actions else None


def optimal_heuristic_policy(obs: int) -> int:
    taxi_row, taxi_col, pass_loc, dest = decode_state(obs)
    target = LOCS[pass_loc] if pass_loc < 4 else LOCS[dest]
    if (taxi_row, taxi_col) == target:
        return 4 if pass_loc < 4 else 5
    return _bfs((taxi_row, taxi_col), target)
