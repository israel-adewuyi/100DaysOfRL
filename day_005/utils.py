import numpy as np
from typing import List

Arr = np.ndarray

def sum_rewards(rewards: List[int], gamma: float = 1):
    """
    Computes the total discounted sum of rewards for an episode.
    By default, assume no discount
    Input:
        rewards [r1, r2, r3, ...] The rewards obtained during an episode
        gamma: Discount factor
    Output:
        The sum of discounted rewards
        r1 + gamma*r2 + gamma^2 r3 + ...
    """
    total_reward = 0
    for r in rewards[:0:-1]:  # reverse, excluding first
        total_reward += r
        total_reward *= gamma
    total_reward += rewards[0]
    return total_reward


def cummean(arr: Arr):
    """
    Computes the cumulative mean
    """
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)


import plotly.graph_objects as go

def plot_returns(returns_dict, env_name, file_path, file_format='png'):
    """
    Plots the average rewards and saves the plot to a file.

    Parameters:
    - returns_dict: Dictionary containing the returns for each agent.
    - env_name: Name of the environment.
    - file_path: Path where the plot will be saved.
    - file_format: Format of the file to save ('png' or 'html').
    """
    fig = go.Figure()

    for agent_name, returns in returns_dict.items():
        fig.add_trace(go.Scatter(x=list(range(len(returns))), y=returns, mode='lines', name=agent_name))

    fig.update_layout(
        title=f"Avg. reward on {env_name}",
        xaxis_title="Episode",
        yaxis_title="Avg. reward",
        template="simple_white",
        width=700,
        height=400,
    )

    if file_format == 'png':
        fig.write_image(file_path)
    elif file_format == 'html':
        fig.write_html(file_path)
    else:
        raise ValueError("Unsupported file format. Use 'png' or 'html'.")