import numpy as np
import gymnasium as gym

from tqdm import tqdm
from utils import plot_rewards
from multi_armed_bandit_env import MultiArmedBandit
from agent import Agent, RandomAgent, EpsilonGreedyAgent

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


# def run_episode(env: gym.Env, agent: Agent):
#     rewards = []

#     truncated = False
    
#     while not truncated:
#         action = agent.get_action()
#         obs, reward, truncated, info = env.step(action)
#         agent.observe(action, reward, indo)
#         rewards.append(reward)




if __name__ == "__main__":
    num_arms = 10
    max_episode_steps = 2_000
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
    
    # print(f"Expected correct freq: {1/10}, actual: {corrects.mean():.6f}")
    # assert np.isclose(corrects.mean(), 1/10, atol=0.05), "Random agent is not random enough!"
    
    # print(f"Expected average reward: 0.0, actual: {rewards.mean():.6f}")
    # assert np.isclose(rewards.mean(), 0, atol=0.05), "Random agent should be getting mean arm reward, which is zero."
    for epsilon in [0.1, 0.01]:
        for initial_value in [0, 5]:
            agent = EpsilonGreedyAgent(num_arms, 0, epsilon, initial_value)
            rewards, corrects = run_agent(env, agent)
            all_rewards.append(rewards)
            names.append(str(agent))
            print(agent)
            print(f" -> Frequency of correct arm: {corrects.mean():.4f}")
            print(f" -> Average reward: {rewards.mean():.4f}")

    # names = ["Random Agent"]
    plot_rewards(all_rewards, names)

    
        