import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym

from tqdm import tqdm
from day_001.utils import plot_rewards
from day_001.multi_armed_bandit_env import MultiArmedBandit
from day_001.agent import Agent, RandomAgent, EpsilonGreedyAgent
from agent import CheaterAgent, UCBSelectionAgent

def run_episode(env: gym.Env, agent: Agent, seed: int):
    '''
    Runs a single episode of interaction between an agent and an environment.

    Args:
        env (gym.Env): The environment in which the agent operates.
        agent (Agent): The agent that takes actions in the environment.
        seed (int): The seed for random number generation to ensure reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing arrays of rewards
        received in each step and a flag indicating if the chosen arm was the best.
    '''
    (rewards, was_best) = ([], [])

    env.reset(seed=seed)
    agent.reset(seed=seed)

    truncated = False
    while not truncated:
        arm = agent.get_action()
        (obs, reward, terminated, truncated, info) = env.step(arm)
        agent.observe(arm, reward, info)
        rewards.append(reward)
        was_best.append(1 if arm == info["best_arm"] else 0)

    rewards = np.array(rewards, dtype=float)
    was_best = np.array(was_best, dtype=int)
    return (rewards, was_best)


def run_agent(env: gym.Env, agent: Agent, n_runs=200, base_seed=1):
    all_rewards = []
    all_was_bests = []
    base_rng = np.random.default_rng(base_seed)
    for n in tqdm(range(n_runs)):
        seed = base_rng.integers(low=0, high=10_000, size=1).item()
        (rewards, corrects) = run_episode(env, agent, seed)
        all_rewards.append(rewards)
        all_was_bests.append(corrects)
    return (np.array(all_rewards), np.array(all_was_bests))


if __name__ == "__main__":
    num_arms = 10
    max_episode_steps = 1_000
    agent = RandomAgent(num_arms, 0)
    # env = MultiArmedBandit(num_arms)

    gym.envs.registration.register(
        id="ArmedBanditTestbed-v0",
        entry_point=MultiArmedBandit,
        max_episode_steps=max_episode_steps,
        nondeterministic=True,
        reward_threshold=1.0,
        kwargs={"num_arms": 10},
    )
    

    env = gym.make("ArmedBanditTestbed-v0", num_arms=num_arms)

    all_rewards = []
    names = []
    
    rewards, corrects = run_agent(env, agent)
    all_rewards.append(rewards)
    names = ["Random Agent"]


    ## Cheater Agent
    cheater_agent = CheaterAgent(num_arms, 0)
    cheater_rewards, cheater_corrects = run_agent(env, cheater_agent)
    all_rewards.append(cheater_rewards)
    names.append(str(cheater_agent))

    # Upper Bound Selection Agent
    UCBSAgent = UCBSelectionAgent(num_arms, 0, 2)
    UCBSrewards, UCBS_corrects = run_agent(env, UCBSAgent)
    all_rewards.append(UCBSrewards)
    names.append(str(UCBSAgent))
    
    for initial_value in [0, 5]:
        agent = EpsilonGreedyAgent(num_arms, 0, 0.1, initial_value)
        rewards, corrects = run_agent(env, agent)
        all_rewards.append(rewards)
        names.append(str(agent))
        print(agent)
        print(f" -> Frequency of correct arm: {corrects.mean():.4f}")
        print(f" -> Average reward: {rewards.mean():.4f}")

    plot_rewards(all_rewards, names, "day_002/artefacts/average_rewards.png")

    
        