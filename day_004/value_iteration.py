import numpy as np

from environment import Environment


def value_iteration_loop(env: Environment, gamma: float):
    """
        Computes the optimal policy by computing the optimal value function with Bellman optimality equations and 
            using the optimal value function V_*(s) to compute the optimal policy, pi(s). 

        Args: 
            env - The MDP environment on which we are running the value iteration
            gamma - The discount factor

        Returns:
            policy: (num_states) - The optimal policy
    """
    num_states = env.num_states
    num_actions = env.num_actions

    env_rewards = env.R
    transitions = env.T

    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)

    while True:
        cur_V = np.zeros(num_states)
        
        for state in range(num_states):
            Q = np.zeros(num_actions)
            for action in range(num_actions):
                expected_reward = np.dot(transitions[state, action], env_rewards[state, action] + gamma * V)
                Q[action] = expected_reward

            cur_V[state] = max(Q)

        if np.allclose(cur_V, V):
            break

        V = cur_V.copy()

    for state in range(num_states):
        Q = np.zeros(num_actions)

        for action in range(num_actions):
            Q[action] = np.dot(transitions[state, action], env_rewards[state, action] + gamma * V)

        policy[state] = int(np.argmax(Q))


    return policy.astype(int)
        

    
            