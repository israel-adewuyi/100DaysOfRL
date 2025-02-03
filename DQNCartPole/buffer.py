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
    obs: Tensor["batch_size obs_shape"],
    act: Tensor["batch_size obs_shape"],
    reward: Tensor["batch_size obs_shape"],
    obs: Tensor["batch_size obs_shape"]
    # there should be another arg for when we hit time limit with Gym envs. 
    # TODO: Think about how to incorporate done/truncated/terminated here. I am not too sure now. 



class Buffer:

    def add(self, samples: BufferSamples):
        pass
    def sample(self,) -> BufferSamples: 
        pass