import torch
import numpy as np

from torch import Tensor
from typing import Tuple
from jaxtyping import Float, Int, Bool
"""
    We want to store experiences here. 

    So I can have Buffer class and BufferSamples class

    Buffer will `add` BufferSamples to buffer,
        but BufferSamples is what is interacting with the env??? Sure??? Now, probably not, at least not on it's own.

    Env -> observation, agent -> action, env -> reward_t+1, obs_t+1. 
    Some other class collects (obs, actin, reward_t+1, obs_t+1) into a BufferSample and calls add. Does this make sense? 
"""

class BufferSamples:
    """
        IDK yet. 

        Args: 

        
    """
    obs: Float[Tensor, "batch_size obs_shape"]
    act: Float[Tensor, "batch_size act_shape"]
    reward: Float[Tensor, "batch_size"]
    terminated: Float[Tensor, "batch_size"]
    next_obs: Float[Tensor, "batch_size obs_shape"]



class Buffer:
    """
    Contains buffer; has a method to sample from it to return a ReplayBufferSamples object.
    """

    rng: np.random.Generator
    obs: Float[np.ndarray, "buffer_size *obs_shape"]
    actions: Float[np.ndarray, "buffer_size *action_shape"]
    rewards: Float[np.ndarray, "buffer_size"]
    terminated: Bool[np.ndarray, "buffer_size"]
    next_obs: Float[np.ndarray, "buffer_size *obs_shape"]

    def __init__(self, num_envs: Int, obs_shape: Tuple[Int], action_shape: Tuple[Int], buffer_size: Int, seed: Int):
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.rng = np.random.default_rng(seed)

        self.obs = np.empty((0, *self.obs_shape), dtype=np.float32)
        self.actions = np.empty((0, *self.action_shape), dtype=np.int32)
        self.rewards = np.empty(0, dtype=np.float32)
        self.terminated = np.empty(0, dtype=bool)
        self.next_obs = np.empty((0, *self.obs_shape), dtype=np.float32)

    def add(
        self,
        obs: Float[np.ndarray, "num_envs *obs_shape"],
        actions: Int[np.ndarray, "num_envs *action_shape"],
        rewards: Float[np.ndarray, "num_envs"],
        terminated: Bool[np.ndarray, "num_envs"],
        next_obs: Float[np.ndarray, "num_envs *obs_shape"],
    ) -> None:
        """
        Add a batch of transitions to the replay buffer.
        """
        
        for data, expected_shape in zip(
            [obs, actions, rewards, terminated, next_obs], [self.obs_shape, self.action_shape, (), (), self.obs_shape]
        ):
            assert isinstance(data, np.ndarray)
            assert data.shape == (self.num_envs, *expected_shape)

        self.obs = np.concatenate((self.obs, obs))[-self.buffer_size :]
        self.actions = np.concatenate((self.actions, actions))[-self.buffer_size :]
        self.rewards = np.concatenate((self.rewards, rewards))[-self.buffer_size :]
        self.terminated = np.concatenate((self.terminated, terminated))[-self.buffer_size :]
        self.next_obs = np.concatenate((self.next_obs, next_obs))[-self.buffer_size :]

    def sample(self, sample_size: Int, device: torch.device) -> BufferSamples:
        """
        Sample a batch of transitions from the buffer, with replacement.
        """
        indices = self.rng.integers(0, self.buffer_size, sample_size)

        return ReplayBufferSamples(
            *[
                t.tensor(x[indices], device=device)
                for x in [self.obs, self.actions, self.rewards, self.terminated, self.next_obs]
            ]
        )