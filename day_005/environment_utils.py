import numpy as np

Arr = np.ndarray
ObsType = int
ActType = int

class Norvig(Environment):
    def dynamics(self, state: int, action: int) -> tuple[Arr, Arr, Arr]:
        def state_index(state):
            assert 0 <= state[0] < self.width and 0 <= state[1] < self.height, print(state)
            pos = state[0] + state[1] * self.width
            assert 0 <= pos < self.num_states, print(state, pos)
            return pos

        pos = self.states[state]
        if state in self.terminal or state in self.walls:
            return (np.array([state]), np.array([0]), np.array([1]))
        out_probs = np.zeros(self.num_actions) + 0.1
        out_probs[action] = 0.7
        out_states = np.zeros(self.num_actions, dtype=int) + self.num_actions
        out_rewards = np.zeros(self.num_actions) + self.penalty
        new_states = [pos + x for x in self.actions]
        for i, s_new in enumerate(new_states):
            if not (0 <= s_new[0] < self.width and 0 <= s_new[1] < self.height):
                out_states[i] = state
                continue
            new_state = state_index(s_new)
            if new_state in self.walls:
                out_states[i] = state
            else:
                out_states[i] = new_state
            for idx in range(len(self.terminal)):
                if new_state == self.terminal[idx]:
                    out_rewards[i] = self.goal_rewards[idx]
        return (out_states, out_rewards, out_probs)

    def render(self, pi: Arr):
        assert len(pi) == self.num_states
        emoji = ["⬆️", "➡️", "⬇️", "⬅️"]
        grid = [emoji[act] for act in pi]
        grid[3] = "🟩"
        grid[7] = "🟥"
        grid[5] = "⬛"
        print("".join(grid[0:4]) + "\n" + "".join(grid[4:8]) + "\n" + "".join(grid[8:]))

    def __init__(self, penalty=-0.04):
        self.height = 3
        self.width = 4
        self.penalty = penalty
        num_states = self.height * self.width
        num_actions = 4
        self.states = np.array([[x, y] for y in range(self.height) for x in range(self.width)])
        self.actions = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        self.dim = (self.height, self.width)
        terminal = np.array([3, 7], dtype=int)
        self.walls = np.array([5], dtype=int)
        self.goal_rewards = np.array([1.0, -1])
        super().__init__(num_states, num_actions, start=8, terminal=terminal)


class Environment:
    def __init__(self, num_states: int, num_actions: int, start=0, terminal=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.start = start
        self.terminal = np.array([], dtype=int) if terminal is None else terminal
        (self.T, self.R) = self.build()

    def build(self):
        """
        Constructs the T and R tensors from the dynamics of the environment.

        Returns:
            T : (num_states, num_actions, num_states) State transition probabilities
            R : (num_states, num_actions, num_states) Reward function
        """
        num_states = self.num_states
        num_actions = self.num_actions
        T = np.zeros((num_states, num_actions, num_states))
        R = np.zeros((num_states, num_actions, num_states))
        for s in range(num_states):
            for a in range(num_actions):
                (states, rewards, probs) = self.dynamics(s, a)
                (all_s, all_r, all_p) = self.out_pad(states, rewards, probs)
                T[s, a, all_s] = all_p
                R[s, a, all_s] = all_r
        return (T, R)

    def dynamics(self, state: int, action: int) -> tuple[Arr, Arr, Arr]:
        """
        Computes the distribution over possible outcomes for a given state
        and action.

        Args:
            state  : int (index of state)
            action : int (index of action)

        Returns:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair
        """
        raise NotImplementedError()

    def render(pi: Arr):
        """
        Takes a policy pi, and draws an image of the behavior of that policy, if applicable.

        Args:
            pi : (num_actions,) a policy

        Returns:
            None
        """
        raise NotImplementedError()

    def out_pad(self, states: Arr, rewards: Arr, probs: Arr):
        """
        Args:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair

        Returns:
            states  : (num_states,) all the next states
            rewards : (num_states,) rewards for each next state transition
            probs   : (num_states,) likelihood of each state-reward pair (including zero-prob outcomes.)
        """
        out_s = np.arange(self.num_states)
        out_r = np.zeros(self.num_states)
        out_p = np.zeros(self.num_states)
        for i in range(len(states)):
            idx = states[i]
            out_r[idx] += rewards[i]
            out_p[idx] += probs[i]
        return out_s, out_r, out_p


class DiscreteEnviroGym(gym.Env):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete
    '''
    A discrete environment class for reinforcement learning, compatible with OpenAI Gym.

    This class represents a discrete environment where actions and observations are discrete.
    It is designed to interface with a provided `Environment` object which defines the
    underlying dynamics, states, and actions.

    Attributes:
        action_space (gym.spaces.Discrete): The space of possible actions.
        observation_space (gym.spaces.Discrete): The space of possible observations (states).
        env (Environment): The underlying environment with its own dynamics and properties.
    '''
    def __init__(self, env: Environment):
        super().__init__()
        self.env = env
        self.observation_space = gym.spaces.Discrete(env.num_states)
        self.action_space = gym.spaces.Discrete(env.num_actions)
        self.reset()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        '''
        Execute an action and return the new state, reward, done flag, and additional info.
        The behaviour of this function depends primarily on the dynamics of the underlying
        environment.
        '''
        (states, rewards, probs) = self.env.dynamics(self.pos, action)
        idx = self.np_random.choice(len(states), p=probs)
        (new_state, reward) = (states[idx], rewards[idx])
        self.pos = new_state
        done = self.pos in self.env.terminal
        return (new_state, reward, done, {"env": self.env})

    def reset(self, seed: int | None = None, options=None) -> ObsType:
        '''
        Resets the environment to its initial state.
        '''
        super().reset(seed=seed)
        self.pos = self.env.start
        return self.pos

    def render(self, mode="human"):
        assert mode == "human", f"Mode {mode} not supported!"