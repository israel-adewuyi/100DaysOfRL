import numpy as np


ActType = int

class Agent:
    '''
    Base class for agents in a multi-armed bandit environment
    '''
    rng: np.random.Generator

    def __init__(self, num_arms: int, seed: int):
        self.num_arms = num_arms
        self.reset(seed)

    def get_action(self) -> ActType:
        raise NotImplementedError()

    def observe(self, action: ActType, reward: float, info: dict) -> None:
        pass

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)




class RandomAgent(Agent):
    '''
        This agent takes actions randomly i.e randomly select an integer between 0 and num_arms.
    '''
    def get_action(self) -> ActType:
        action  = np.random.randint(0, self.num_arms)
        return action


class EpsilonGreedyAgent(Agent):
    '''
        This agent takes action based on the epsilon greedy action selection method. 
    '''
    def __init__(self, num_arms: int, seed: int, epsilon: float, initial_value: int):
        self.initial_value = initial_value
        self.epsilon = epsilon

        super().__init__(num_arms, seed)

    def get_action(self) -> ActType:
        probability = self.rng.random()

        if probability < self.epsilon:
            action = self.rng.integers(0, self.num_arms)
        else:
            action = int(np.argmax(self.Q))

        return action

    def observe(self, action: ActType, reward: float, info: dict) -> None:
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.Q = np.full(self.num_arms, self.initial_value, dtype=float)
        self.N = np.zeros(self.num_arms)

    def __repr__(self):
        return f"EpsilonGreedyAgent_eps={self.epsilon}_initial_value={self.initial_value})"