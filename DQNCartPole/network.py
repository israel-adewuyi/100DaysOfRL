import torch.nn as nn
from torch import Tensor

from typing import List, Tuple


class QNetwork(nn.Module):
    """
        A fully connected network, this class represents the policy network for the DQN on the Cart Pole 
        environment (https://gymnasium.farama.org/environments/classic_control/cart_pole/).
        

        Args: 
        num_actions: int:
            the number of possible actions the agent can take. In the case of cart polem left(0) or right(1)
        layers: nn.Sequential
            the layers of the fully connected network used as the policy network
    
        Methods: 
        init(observation_shape: Tuple[int], num_actions: int, hidden_states: List[int]):
            initializes a FCN that will serve as the policy network
        foward(x: Tensor[float]):
            runs forward pass on the input tensor, which is an observation from the cart pole environment
        
    """

    def __init__(self, num_actions: int, obs_shape: Tuple[int], hidden_states: List[int] = [124, 80]):
        super().__init__()
        self.num_actions = num_actions
        
        self.layers = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_states[0]),
            nn.ReLU(),
            nn.Linear(hidden_states[0], hidden_states[1]),
            nn.ReLU(),
            nn.Linear(hidden_states[1], self.num_actions)
        )
        


    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)