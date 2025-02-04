import numpy as np

from network import QNetwork
from buffer import Buffer
from jaxtyping import Float, Int
from gymnasium.vector import SyncVectorEnv
from utils import linear_scheduler, epsilon_greedy_action_selection


class DQNAgent:
    rng = np.random.Generator
    def __init__(self,
                 q_network: QNetwork,
                 buffer: Buffer,
                 envs: SyncVectorEnv,
                 start_eps: Float,
                 end_eps: Float, 
                 exploration_fraction: Float,
                 total_time_steps: int,
                 rng: np.random.Generator
    ):
        self.q_network = q_network
        self.buffer = buffer
        self.envs = envs
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.exploration_fraction = exploration_fraction
        self.total_time_steps = total_time_steps
        self.rng = rng

        self.steps = 0
        self.obs, _ = self.envs.reset()
        self.epsilon = start_eps

    def agent_step(self,) -> dict:
        # We want to fill up the buffer
        obs = self.obs
        actions = self.get_action(obs)

        next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)

        actual_next_obs = next_obs.copy()

        for idx in self.envs.num_envs:
            if (terminated | truncated)[n]:
                actual_next_obs = infos['final_observation'][n]


        self.buffer.add(obs, actions, rewards, terminated, actual_next_obs)
        self.obs = next_obs
        self.steps += self.envs.num_envs
        
        return infos

    def get_action(self, obs: np.ndarray): # I am not really sure obs is of type numpy array
        epsilon = linear_scheduler(self.start_eps, self.end_eps, self.steps, self.exploration_fraction, self.total_time_steps)
        actions = epsilon_greedy_action_selection(obs , self.q_network, self.envs, epsilon, self.rng)