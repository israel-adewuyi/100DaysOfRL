import time
import wandb
import torch
import numpy as np
import gymnasium as gym

from tqdm import tqdm
from torch import Tensor
from buffer import Buffer
from network import QNetwork
from agent import DQNAgent
from dataclasses import dataclass
from jaxtyping import Bool, Int, Float
from typing import Optional, Union
from utils import get_episode_data_from_infos, set_global_seeds, make_env

@dataclass
class DQNArgs:
    # Basic / global
    seed: int = 1
    env_id: str = "CartPole-v1"
    num_envs: int = 1

    # Wandb / logging
    use_wandb: bool = False
    wandb_project_name: str = "DQNCartPole"
    wandb_entity: Optional[str] = None
    video_log_freq: Optional[int] = 50

    # Duration of different phases / buffer memory settings
    total_timesteps: int = 500_000
    steps_per_train: int = 10
    trains_per_target_update: int = 100
    buffer_size: int = 10_000

    # Optimization hparams
    batch_size: int = 128
    learning_rate: float = 2.5e-4

    # RL-specific
    gamma: float = 0.99
    exploration_fraction: float = 0.2
    start_e: float = 1.0
    end_e: float = 0.1

    device = "cuda:3"

    def __post_init__(self):
        assert self.total_timesteps - self.buffer_size >= self.steps_per_train
        self.total_training_steps = (self.total_timesteps - self.buffer_size) // self.steps_per_train
        self.video_save_path = "videos"


# args = DQNArgs(total_timesteps=400_000)  # changing total_timesteps will also change ???
# print(args)
# utils.arg_help(args)

# class DQNTrainer():
#     def __init__(self, args: DQNArgs):
#         set_global_seeds(args.seed)
#         self.args = args
#         self.rng = np.random.default_rng(args.seed)
#         self.run_name = f"{args.env_id}_{args.wandb_project_name}_seed{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
#         self.envs = gym.vector.SyncVectorEnv(
#             [make_env(idx=idx, run_name=self.run_name, **args.__dict__) for idx in range(args.num_envs)]
#         )

#         num_envs = self.envs.num_envs
#         action_shape = self.envs.single_action_space.shape
#         num_actions = self.envs.single_action_space.n
#         obs_shape = self.envs.single_observation_space.shape
#         assert action_shape == ()

#         # Create our replay buffer
#         self.buffer = Buffer(num_envs, obs_shape, action_shape, args.buffer_size, args.seed)

#         # Create our networks & optimizer (target network should be initialized with a copy of the Q-network's weights)
#         self.q_network = QNetwork(num_actions, obs_shape).to(self.args.device)
#         self.target_network = QNetwork(num_actions, obs_shape).to(self.args.device)
#         self.target_network.load_state_dict(self.q_network.state_dict())
#         self.optimizer = torch.optim.AdamW(self.q_network.parameters(), lr=args.learning_rate)

#         # Create our agent
#         self.agent = DQNAgent(
#             self.q_network,
#             self.buffer,
#             self.envs,
#             args.start_e,
#             args.end_e,
#             args.exploration_fraction,
#             args.total_timesteps,
#             self.rng,
#         )

#         print("DQNTrainer QNetwork ID:", id(self.q_network))
#         print("DQNTrainer Buffer ID:", id(self.buffer))

#         print("Details of environment")
#         print(self.envs)
#         print("Action space")
#         print(self.envs.action_space)

#     def add_to_replay_buffer(self, n: int, verbose: Bool = False):
#         data = None
#         t0 = time.time()

#         for step in tqdm(range(n), disable=not verbose, desc="Adding to replay buffer"):
#             print("Got here 0")
#             infos = self.agent.agent_step()

#             # Get data from environments, and log it if some environment did actually terminate
#             new_data = get_episode_data_from_infos(infos)
#             if new_data is not None:
#                 data = new_data  # makes sure we return a non-empty dict at the end, if some episode terminates
#                 if self.args.use_wandb:
#                     wandb.log(new_data, step=self.agent.step)

#         # Log SPS
#         if self.args.use_wandb:
#             wandb.log({"SPS": (n * self.envs.num_envs) / (time.time() - t0)}, step=self.agent.steps)

#         return data
        
#     def prepopulate_replay_buffer(self, ):
#         """
#         Called to fill the replay buffer before training starts.
#         """
#         n_steps_to_fill_buffer = self.args.buffer_size // self.args.num_envs
#         self.add_to_replay_buffer(n_steps_to_fill_buffer, verbose=True)
        
#     def training_step(self, step: Int):
#         data = self.buffer.sample(self.args.batch_size, self.args.device) 

#         with torch.inference_mode():
#             target_max = self.target_network(data.next_obs).max(-1).values
#         predicted_q_vals = self.q_network(data.obs)[range(len(data.act)), data.act]

#         td_error = data.reward + self.args.gamma * target_max * (1 - data.terminated.float()) - predicted_q_vals
#         loss = td_error.pow(2).mean()
#         loss.backward()
#         self.optimizer.step()
#         self.optimizer.zero_grad()

#         if step % self.args.trains_per_target_update == 0:
#             self.target_network.load_state_dict(self.q_network.state_dict())

#         if self.args.use_wandb:
#             wandb.log(
#                 {"td_loss": loss, "q_values": predicted_q_vals.mean().item(), "epsilon": self.agent.epsilon, "rewards": data.reward.mean().item()},
#                 step=self.agent.steps,
#             )
        
#     def train(self) -> None:
#         if self.args.use_wandb:
#             wandb.init(
#                 project=self.args.wandb_project_name,
#                 entity=self.args.wandb_entity,
#                 name=self.run_name,
#                 monitor_gym=self.args.video_log_freq is not None,
#             )
#             wandb.watch(self.q_network, log="all", log_freq=50)

#         self.prepopulate_replay_buffer()

#         pbar = tqdm(range(self.args.total_training_steps))
#         last_logged_time = time.time()  # so we don't update the progress bar too much

#         for step in pbar:
#             data = self.add_to_replay_buffer(self.args.steps_per_train)
#             if data is not None and time.time() - last_logged_time > 0.5:
#                 last_logged_time = time.time()
#                 pbar.set_postfix(**data)

#             self.training_step(step)

#         self.envs.close()
#         if self.args.use_wandb:
#             wandb.finish()

class DQNTrainer:
    def __init__(self, args: DQNArgs):
        set_global_seeds(args.seed)
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.run_name = f"{args.env_id}__{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(idx=idx, run_name=self.run_name, **args.__dict__) for idx in range(args.num_envs)]
        )

        # Define some basic variables from our environment (note, we assume a single discrete action space)
        num_envs = self.envs.num_envs
        action_shape = self.envs.single_action_space.shape
        num_actions = self.envs.single_action_space.n
        obs_shape = self.envs.single_observation_space.shape
        assert action_shape == ()

        # Create our replay buffer
        self.buffer = Buffer(num_envs, obs_shape, action_shape, args.buffer_size, args.seed)

        # Create our networks & optimizer (target network should be initialized with a copy of the Q-network's weights)
        self.q_network = QNetwork(num_actions, obs_shape).to(args.device)
        self.target_network = QNetwork(num_actions, obs_shape).to(args.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.AdamW(self.q_network.parameters(), lr=args.learning_rate)

        # Create our agent
        self.agent = DQNAgent(
            self.envs,
            self.buffer,
            self.q_network,
            args.start_e,
            args.end_e,
            args.exploration_fraction,
            args.total_timesteps,
            self.rng,
        )
    def add_to_replay_buffer(self, n: int, verbose: bool = False):
        """
        Takes n steps with the agent, adding to the replay buffer (and logging any results). Should return a dict of
        data from the last terminated episode, if any.

        Optional argument `verbose`: if True, we can use a progress bar (useful to check how long the initial buffer
        filling is taking).
        """
        data = None
        t0 = time.time()

        for step in tqdm(range(n), disable=not verbose, desc="Adding to replay buffer"):
            infos = self.agent.play_step()
            print("Got here 0")
            # Get data from environments, and log it if some environment did actually terminate
            new_data = get_episode_data_from_infos(infos)
            if new_data is not None:
                data = new_data  # makes sure we return a non-empty dict at the end, if some episode terminates
                if self.args.use_wandb:
                    wandb.log(new_data, step=self.agent.step)

        # Log SPS
        if self.args.use_wandb:
            wandb.log({"SPS": (n * self.envs.num_envs) / (time.time() - t0)}, step=self.agent.step)

        return data

    def prepopulate_replay_buffer(self):
        """
        Called to fill the replay buffer before training starts.
        """
        n_steps_to_fill_buffer = self.args.buffer_size // self.args.num_envs
        self.add_to_replay_buffer(n_steps_to_fill_buffer, verbose=True)

    def training_step(self, step: int) -> Float[Tensor, ""]:
        """
        Samples once from the replay buffer, and takes a single training step. The `step` argument is used to track the
        number of training steps taken.
        """
        data = self.buffer.sample(self.args.batch_size, device) 

        with t.inference_mode():
            target_max = self.target_network(data.next_obs).max(-1).values
        predicted_q_vals = self.q_network(data.obs)[range(len(data.act)), data.act]

        td_error = data.reward + self.args.gamma * target_max * (1 - data.terminated.float()) - predicted_q_vals
        loss = td_error.pow(2).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if step % self.args.trains_per_target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.args.use_wandb:
            wandb.log(
                {"td_loss": loss, "q_values": predicted_q_vals.mean().item(), "epsilon": self.agent.epsilon, "rewards": data.rewards.mean().item(), "rewards": data.reward.mean().item()},
                step=self.agent.step,
            )


    def train(self) -> None:
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
                monitor_gym=self.args.video_log_freq is not None,
            )
            wandb.watch(self.q_network, log="all", log_freq=50)

        self.prepopulate_replay_buffer()

        pbar = tqdm(range(self.args.total_training_steps))
        last_logged_time = time.time()  # so we don't update the progress bar too much

        for step in pbar:
            data = self.add_to_replay_buffer(self.args.steps_per_train)
            if data is not None and time.time() - last_logged_time > 0.5:
                last_logged_time = time.time()
                pbar.set_postfix(**data)

            self.training_step(step)

        self.envs.close()
        if self.args.use_wandb:
            wandb.finish()


# if __name__ == "__main__":
#     args = DQNArgs(use_wandb=True)
#     trainer = DQNTrainer(args)
#     trainer.train()











        