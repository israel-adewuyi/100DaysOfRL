import numpy as np
import plotly.graph_objects as go

from typing import List, Union

CONFIG = dict(displayModeBar=False)

def moving_avg(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_rewards(
    all_rewards: List[np.ndarray],
    names: List[str],
    filename: str,
    moving_avg_window: Union[int, None] = 15,
):
    names = [name.replace(", ", "<br>  │").replace("(", "<br>  │").replace(")", "") for name in names]

    fig = go.Figure(layout=dict(template="simple_white", title_text="Mean reward over all runs"))
    for rewards, name in zip(all_rewards, names):
        rewards_avg = rewards.mean(axis=0)
        if moving_avg_window is not None:
            rewards_avg = moving_avg(rewards_avg, moving_avg_window)
        fig.add_trace(go.Scatter(y=rewards_avg, mode="lines", name=name))
        fig.update_layout(height=500, width=900).show(config=CONFIG)
    
    fig.write_image(filename)
    print("Average rewards written to file")