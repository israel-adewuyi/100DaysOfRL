import numpy as np
from day_001.agent import Agent

ActType = int

class CheaterAgent(Agent):
    def __init__(self, num_arms: int, seed: int):
        super().__init__(num_arms, seed)
        self.best_arm = 0

    def observe(self, action: ActType, reward: float, info: dict) -> ActType:
        self.best_arm = info["best_arm"]

    def get_action(self) -> ActType:
        return self.best_arm

    def reset(self, seed: int):
        super().reset(seed)

    def __repr__(self, ):
        return "CheaterAgent"


class UCBSelectionAgent(Agent):
    def __init__(self, num_arms: int, seed: int, c: float):
        super().__init__(num_arms, seed)
        self.c = c

    def observe(self, action: ActType, reward: float, info: dict) -> None:
        self.t += 1
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]        

    def get_action(self) -> ActType:
        action = int(np.argmax(self.Q + self.c * np.sqrt(np.log2(self.t) / (self.N + 1e-8))))
        return action

    def reset(self, seed: int):
        super().reset(seed)
        self.t = 1
        self.Q = np.zeros(self.num_arms)
        self.N = np.zeros(self.num_arms)

    def __repr__(self, ):
        return f"UCB_c={self.c}"