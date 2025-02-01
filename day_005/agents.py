import numpy as np
from typing import List
from .environment_utils import EpsilonGreedy, Experience, Agent

ObsType = int
ActType = int

class SARSA(EpsilonGreedy):
    def observe(self, exp: Experience):
        obs_old, act_old, reward, obs_new, act_new = exp.obs, exp.act, exp.reward, exp.new_obs, exp.new_act

        self.Q[obs_old, act_old] += self.config.lr * (reward + (self.gamma * self.Q[obs_new, act_new]) - self.Q[obs_old, act_old])

    def run_episode(self, seed) -> List[int]:
        obs, info = self.env.reset(seed=seed)
        act = self.get_action(obs)
        terminated = False
        rewards = []
        
        while not terminated:
            (obs_new, reward, truncated, terminated, info) = self.env.step(act)
            act_new = self.get_action(obs_new)
            exp = Experience(obs, act, reward, obs_new, act_new)
            self.observe(exp)
            rewards.append(reward)
            obs = obs_new
            act = act_new

        return rewards


class Random(Agent):
    def get_action(self, obs: ObsType) -> ActType:
        return self.rng.integers(0, self.num_actions)


class QLearning(EpsilonGreedy):
    def observe(self, exp: Experience) -> None:
        obs_old, act_old, reward, obs_new, act_new = exp.obs, exp.act, exp.reward, exp.new_obs, exp.new_act

        self.Q[obs_old, act_old] += self.config.lr * (reward + (self.gamma * np.max(self.Q[obs_new])) - self.Q[obs_old, act_old])