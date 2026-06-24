import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorld(gym.Env):
    def __init__(self, size=20, goal_reward=1.0, step_penalty=-0.01, slip_prob=0.1):
        super().__init__()
        self.size = size
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        
        # Discrete state: row * size + col
        self.observation_space = spaces.Discrete(size * size)
        # 4 actions: up, down, left, right
        self.action_space = spaces.Discrete(4)
        
        # Goal is bottom right
        self.goal = (size-1, size-1)
        self.pos = (0, 0)
        
        self.slip_prob = slip_prob
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # this seeds self.np_random
        # Random start position
        self.pos = (
            np.random.randint(0, self.size),
            np.random.randint(0, self.size)
        )
        return self._state(), {}
    
    def _state(self):
        return self.pos[0] * self.size + self.pos[1]
    
    def step(self, action):
        if np.random.random() < self.slip_prob:
            action = self.action_space.sample()
        r, c = self.pos
        if action == 0: r = max(0, r-1)        # up
        elif action == 1: r = min(self.size-1, r+1)  # down
        elif action == 2: c = max(0, c-1)        # left
        elif action == 3: c = min(self.size-1, c+1)  # right
        
        self.pos = (r, c)
        done = self.pos == self.goal
        reward = self.goal_reward if done else self.step_penalty
        
        return self._state(), reward, done, False, {}

    def render(self):
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        grid[self.goal[0]][self.goal[1]] = 'G'
        grid[self.pos[0]][self.pos[1]] = 'A'
        print('\n'.join([''.join(row) for row in grid]))