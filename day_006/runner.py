import os
import sys
import gymnasium as gym
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from day_005.agents import SARSA, Random, QLearning
from day_005.utils import cummean, plot_returns
from day_005.environment_utils import AgentConfig, DiscreteEnviroGym, Agent
from day_003.gridworld import Norvig



if __name__ == "__main__":
    n_runs = 1000
    gamma = 0.99
    seed = 1


    gym.envs.registration.register(
        id="NorvigGrid-v0",
        entry_point=DiscreteEnviroGym,
        max_episode_steps=100,
        nondeterministic=True,
        kwargs={"env": Norvig(penalty=-0.04)},
    )
    
    env_norvig = gym.make("NorvigGrid-v0")
    config_norvig = AgentConfig()
    args_norvig = (env_norvig, config_norvig, gamma, seed)
    agents_norvig: List[Agent] = [
        # Cheater(*args_norvig),
        QLearning(*args_norvig),
        SARSA(*args_norvig),
        Random(*args_norvig),
    ]
    returns_dict = {}
    for agent in agents_norvig:
        returns = agent.train(n_runs)
        returns_dict[agent.name] = cummean(returns)

    plot_returns(returns_dict, env_norvig.spec.name, "day_006/avg_rewards.png")