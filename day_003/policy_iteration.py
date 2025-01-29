import einops
import numpy as np

from environment import Environment

def policy_evaluation(policy: np.ndarray, env: Environment, gamma: float, eps: float = 1e-8, max_iterations: int = 10_000):
    num_states = env.num_states
    V = np.zeros(num_states)
    rewards = env.R
    transitions = env.T
    
    for i in range(max_iterations):
        cur_V = V.copy()
        for s in range(num_states):
            action = policy[s]
            
            reward = rewards[s, action]

            assert (reward.shape == (num_states, ))
            
            new_state = transitions[s, action]

            cur_V[s] = np.dot(new_state, reward + gamma * V)
            
        if max(abs(cur_V - V)) < eps:
            break
        V = cur_V.copy()

    return V

def policy_improvement(env: Environment, V: np.ndarray, gamma: float):
    num_states, num_actions = env.num_states, env.num_actions
    policy_stable = True
    env_rewards = env.R
    transitions = env.T
    new_policy = np.zeros(num_states)

    for state in range(num_states):
        cur_V = np.zeros(num_actions)

        for action in range(num_actions):
            action_rewards = env_rewards[state, action]
            
            transition_probs = transitions[state, action]

            cur_V[action] = np.dot(transition_probs, action_rewards + gamma * V)

        new_policy[state] = np.argmax(cur_V)

    return new_policy.astype(int)


def policy_iteration_loop(env: Environment, gamma: float, max_iterations: int = 10_000):
    policy = np.zeros(env.num_states, dtype=int)

    for i in range(max_iterations):
        V = policy_evaluation(policy, env, gamma)

        new_policy = policy_improvement(env, V, gamma)

        if np.allclose(policy, new_policy):
            break

        policy = new_policy.copy()

    return policy        