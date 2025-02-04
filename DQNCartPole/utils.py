import random
import torch
import numpy as np
import gymnasium as gym

from jaxtyping import Float, Int
from typing import Dict, Union, Optional
from network import QNetwork
from gymnasium.vector import SyncVectorEnv


def epsilon_greedy_action_selection(obs: np.ndarray,
                                    q_network: QNetwork,
                                    envs: SyncVectorEnv, 
                                    epsilon: float, 
                                    rng: np.random.Generator
)-> np.ndarray:
    obs = torch.from_numpy(obs).to("cuda:2")
    num_actions = envs.single_action_space.n
    num_envs = obs.shape[0]
    
    if epsilon < rng.random():
        actions = rng.integers(0, num_actions, (num_envs, ))
    else:
        q_values = q_network(obs)
        actions = q_values.argmax(-1).detach().cpu().numpy()

    return actions



def linear_scheduler(
    start_eps: Float,
    end_eps: Float, 
    cur_steps: Int,
    exploration_fraction: Float,
    total_timesteps: Int
) -> Float:
    max_time_steps = exploration_fraction * total_timesteps
    epsilon = start_eps + (end_eps - start_eps) * min(cur_steps / max_time_steps, 1)

    return epsilon


def get_episode_data_from_infos(infos: dict) -> Optional[Dict[str, Union[int, float]]]:
    """
    Helper function: returns dict of data from the first terminated environment, if at least one terminated.
    """
    for final_info in infos.get("final_info", []):
        if final_info is not None and "episode" in final_info:
            return {
                "episode_length": final_info["episode"]["l"].item(),
                "episode_reward": final_info["episode"]["r"].item(),
                "episode_duration": final_info["episode"]["t"].item(),
            }

def set_global_seeds(seed):
    """Sets random seeds in several different ways (to guarantee reproducibility)"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def make_env(
    env_id: str,
    seed: int,
    idx: int,
    run_name: str,
    mode: str = "classic-control",
    video_log_freq: int = None,
    video_save_path: str = None,
    **kwargs,
):
    """
    Return a function that returns an environment after setting up boilerplate.
    """

    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if idx == 0 and video_log_freq:
            env = gym.wrappers.RecordVideo(
                env,
                f"{video_save_path}/{run_name}",
                episode_trigger=lambda episode_id: episode_id % video_log_freq == 0,
                disable_logger=True,
            )

        if mode == "atari":
            env = prepare_atari_env(env)
        elif mode == "mujoco":
            env = prepare_mujoco_env(env)

        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
    