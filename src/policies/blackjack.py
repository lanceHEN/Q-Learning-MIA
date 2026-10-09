"""
Basic strategy policy for the Blackjack-v1 environment.

Observation: (player_sum, dealer_card, usable_ace)
Actions: 0 = stick, 1 = hit

Implements the standard simplified basic strategy:
  - Always hit below 12 (busting is impossible)
  - Soft hand (usable ace): hit on ≤17, stick on ≥18
  - Hard hand: stick on ≥17; for 12–16 defer to dealer card
    (stand against a weak dealer 2–6, hit against a strong dealer 7–A)
"""
import numpy as np
from typing import Callable, Tuple, Union


def blackjack_policy(obs: Union[Tuple[int, int, bool], Tuple[int, int, int]]) -> int:
    """
    Returns the basic-strategy action for the given Blackjack-v1 observation.

    Args:
        obs: (player_sum, dealer_card, usable_ace)

    Returns:
        0 (stick) or 1 (hit)
    """
    player_sum, dealer_card, usable_ace = obs

    if player_sum < 12:
        return 1  # can't bust, always hit

    if usable_ace:
        # Soft hand: hit on soft ≤17, stick on soft ≥18
        return int(player_sum <= 17)
    else:
        # Hard hand
        if player_sum >= 17:
            return 0  # stand
        elif player_sum <= 11:
            return 1  # hit (can't bust)
        else:
            # Hard 12–16: stand against weak dealer (2–6), hit against strong (7–A)
            return int(dealer_card >= 7)
