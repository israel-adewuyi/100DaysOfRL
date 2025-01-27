from typing import Optional

import numpy as np
import gymnasium as gym

ActType = int
ObsType = int
class MultiArmedBandit(gym.Env):
    action_space: ActType
    observation_space: ObsType
    rewards: np.ndarray

    
    def __init__(self, num_arms):
        self.action_space = gym.spaces.Discrete(num_arms)
        self.observation_space = gym.spaces.Discrete(1)
        self.num_arms = num_arms
        
        self.reset()

    def step(self, action: ActType):
        if not self.action_space.contains(action):
            return

        reward = np.random.normal(loc = self.rewards[action], scale=1)
        obs = 0
        terminated = False
        truncated = False
        info = {"best_arm":self.best_arm}

        return (obs, reward, terminated, truncated, info)

    def render(self, ):
        pass
    
    def reset(self, seed: Optional[int]=None, options=None):
        self.seed = seed
        # self.rewards = np.random.randn(self.num_arms,)
        self.rewards = self.np_random.normal(loc=0.0, scale=1.0, size=self.num_arms)
        self.best_arm = int(np.argmax(self.rewards))

        obs = 0
        info = {"best_arm":self.best_arm}

        return obs, info
        
    