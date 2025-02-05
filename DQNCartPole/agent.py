import numpy as np
import gymnasium as gym

from pprint import pprint
from network import QNetwork
from buffer import Buffer
from jaxtyping import Float, Int
from gymnasium.vector import SyncVectorEnv
from utils import linear_scheduler, epsilon_greedy_action_selection

class DQNAgent:
    """Base Agent class handling the interaction with the environment."""

    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        buffer: Buffer,
        q_network: QNetwork,
        start_e: Float,
        end_e: Float,
        exploration_fraction: Float,
        total_timesteps: Int,
        rng: np.random.Generator,
    ):
        self.envs = envs
        self.buffer = buffer
        self.q_network = q_network
        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction
        self.total_timesteps = total_timesteps
        self.rng = rng

        self.step = 0  # Tracking number of steps taken (across all environments)
        self.obs, _ = self.envs.reset()  # Need a starting observation
        self.epsilon = start_e  # Starting value (will be updated in `get_actions`)

    def play_step(self) -> dict:
        """
        Carries out a single interaction step between agent & environment, and adds results to the replay buffer.

        Returns `infos` (list of dictionaries containing info we will log).
        """
        obs = self.obs
        # print(type(obs))
        actions = self.get_actions(obs)
        next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)
        
        print(next_obs, actions, rewards)

        # Get `real_next_obs` by finding all environments where we terminated & replacing `next_obs` with the actual terminal states
        # true_next_obs = next_obs.copy()
        # for n in range(self.envs.num_envs):
        #     if terminated[n]:
        #         print(terminated, truncated, infos)
        #         true_next_obs[n] = infos["final_observation"][n]
            
        self.buffer.add(obs, actions, rewards, terminated, next_obs)
        self.obs = next_obs

        self.step += self.envs.num_envs
        return infos

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        """
        Samples actions according to the epsilon-greedy policy using the linear schedule for epsilon.
        """
        """
            To get actions from observations, 
                pass obs to q_network to generate the q_values
                select action with epsilon greedy.
        """
        self.epsilon = linear_scheduler(self.start_e, self.end_e, self.step, self.exploration_fraction, self.total_timesteps)

        actions = epsilon_greedy_action_selection(obs, self.q_network, self.envs, self.epsilon, self.rng)

        return actions

# class DQNAgent:
#     rng = np.random.Generator
#     def __init__(self,
#                  q_network: QNetwork,
#                  buffer: Buffer,
#                  envs: SyncVectorEnv,
#                  start_eps: Float,
#                  end_eps: Float, 
#                  exploration_fraction: Float,
#                  total_time_steps: int,
#                  rng: np.random.Generator
#     ):
#         self.q_network = q_network
#         self.buffer = buffer
#         self.envs = envs
#         self.start_eps = start_eps
#         self.end_eps = end_eps
#         self.exploration_fraction = exploration_fraction
#         self.total_time_steps = total_time_steps
#         self.rng = rng

#         self.steps = 0
#         self.obs, _ = self.envs.reset()
#         self.epsilon = start_eps

#         print("DQNAgent QNetwork ID:", id(self.q_network))
#         print("DQNAgent Buffer ID:", id(self.buffer))

#     def agent_step(self,) -> dict:
#         # We want to fill up the buffer
#         obs = self.obs
#         # print(obs)
#         actions = self.get_action(obs)
        
#         next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)
#         # print("First")
#         # print(next_obs, actions, rewards)
        
#         # rewards = np.array([1])
#         # print("Second")
#         # print(next_obs, actions, rewards)
#         # rewards = [1.0]
#         actual_next_obs = next_obs.copy()

#         # for idx in range(self.envs.num_envs):
#         #     if (terminated | truncated)[idx]:
#         #         print("I am at a terminal spot with ", self.envs.num_envs)
#         #         print(infos)
#         #         pprint(infos)
#         #         print(dir(infos))
#         #         actual_next_obs = infos['next_obs'][idx]


#         self.buffer.add(obs, actions, rewards, terminated, actual_next_obs)
#         self.obs = next_obs
#         self.steps += self.envs.num_envs
        
#         return infos

    # def get_action(self, obs: np.ndarray): # I am not really sure obs is of type numpy array
    #     epsilon = linear_scheduler(self.start_eps, self.end_eps, self.steps, self.exploration_fraction, self.total_time_steps)
    #     actions = epsilon_greedy_action_selection(obs , self.q_network, self.envs, epsilon, self.rng)

    #     return actions