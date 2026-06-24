"""
MiniGrid/BabyAI support for CustomFixedPolicyDataOracle.

Usage:
    env = BabyAIBotEnv(FlatObsWrapper(gym.make("BabyAI-GoToObj-v0")))
    policy = make_minigrid_bot_policy(env)
    oracle = CustomFixedPolicyDataOracle(
        CustomFixedPolicyDataOracleConfig(env=env, action_selector=policy)
    )
"""

import gymnasium as gym
from minigrid.utils.baby_ai_bot import BabyAIBot


class BabyAIBotEnv(gym.Wrapper):
    """
    Gym wrapper that maintains a BabyAIBot and refreshes it on each reset.
    Stack this on top of FlatObsWrapper so the obs are already flat.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._bot = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._bot = BabyAIBot(self.unwrapped)
        return obs, info

    def bot_action(self) -> int:
        try:
            return self._bot.replan()
        except Exception:
            return self.action_space.sample()


def make_minigrid_bot_policy(env: BabyAIBotEnv):
    """
    Returns a callable (obs -> action) for use as action_selector in
    CustomFixedPolicyDataOracleConfig. Ignores the obs arg and drives the
    BabyAIBot via the env's internal state.
    """
    return lambda obs: env.bot_action()
