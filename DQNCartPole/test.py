import torch
import numpy as np
import gymnasium as gym

from DQNTrainer import DQNArgs, DQNTrainer
from typing import Union, Optional
from jaxtyping import Int, Float
from network import QNetwork
from gymnasium.spaces import Discrete, Box

ActType = Int

def test_QNetwork():
    model = QNetwork(2, (4, ), [120, 84])
    param_count = sum(p.nelement() for p in model.parameters())
    assert param_count == 10934


class Probe1(gym.Env):
    action_space: Discrete
    observation_space: Box

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()
        self.action_space = Discrete(1)
        self.observation_space = Box(np.array([0]), np.array([0]))
        self.reset()
    
    def step(self, action):
        # next obs, reward, terminated, truncated, info
        # print("Got here 1")
        return (np.array([0]), 1, True, True, {})

    def reset(self, seed: Union[int, None] = None, options: Union[str, None] = None):
        super().reset(seed=seed)
        return (np.array([0]), {})

    
    


def test_probes(probe_idx):
    gym.envs.registration.register(id=f"Probe{probe_idx}-v0", entry_point=Probe1)
    env = gym.make("Probe1-v0")
    
    args = DQNArgs(
        env_id=f"Probe{probe_idx}-v0",
        wandb_project_name=f"test-probe-{probe_idx}",
        total_timesteps=3000 if probe_idx <= 2 else 5000,
        learning_rate=0.001,
        buffer_size=500,
        use_wandb=True,
        trains_per_target_update=20,
        video_log_freq=None,
    )
    trainer = DQNTrainer(args)
    trainer.train()

    # Get the correct set of observations, and corresponding values we expect
    obs_for_probes = [[[0.0]], [[-1.0], [+1.0]], [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
    expected_value_for_probes = [
        [[1.0]],
        [[-1.0], [+1.0]],
        [[args.gamma], [1.0]],
        [[-1.0, 1.0]],
        [[1.0, -1.0], [-1.0, 1.0]],
    ]
    tolerances = [5e-4, 5e-4, 5e-4, 5e-4, 1e-3]
    obs = torch.tensor(obs_for_probes[probe_idx - 1]).to(args.device)

    # Calculate the actual value, and verify it
    value = trainer.q_network(obs)
    expected_value = torch.tensor(expected_value_for_probes[probe_idx - 1]).to(args.device)
    print(f"Args.gama is {args.gamma}")
    print(f"My Qvalue is {value}")
    print(f"The expected value is {expected_value}")
    torch.testing.assert_close(value, expected_value, atol=tolerances[probe_idx - 1], rtol=0)
    print("Probe tests passed!\n")




if __name__ == "__main__":
    test_QNetwork()
    test_probes(1)
    