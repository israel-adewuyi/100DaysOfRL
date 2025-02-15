I'm working on / learning RL and sharing my progress publicly. 

Library - [Gymnasium](https://gymnasium.farama.org/)

Main reference code : [ARENA RL chapter](https://github.com/callummcdougall/ARENA_3.0/tree/main/chapter2_rl)

# Day 1
- Worked on the Multi-arm bandit problem
- Implemented the epsilon-greedy action value method and compared different epsilon values. From the graph below, it seems they all converge given a long enough episode.
![reward averaging results](/day_001/artefacts/average_rewards.png)
- `day_001/multi_armed_bandit_env.py` - implementation of the multi armed bandit environment
- `day_001/agent.py` -Implementation of the epsilon-greedy agent and random agent
- `day_001/main.py` - code to run the agents
- `day_001/utils.py` and `day_001/main.py` were shamelessly copied from the reference code above, with some slight modifications.

---
# Day 2
- Continued work on the Multi-arm bandit problem from Chapter 2 of the Sutton book.
- Implemented the upper confidence bound selection and compared with an agent that always chooses the best action, sort of a cheating agent. 
![reward averaging results](/day_002/artefacts/average_rewards.png)
- `day_002/agent.py` -Implementation of the Upper Confidenc Bound action selection agent and the cheating agent
- `day_002/main.py` - code to run the agents and compare with the epsilon-greedy agent and the random agent

---
# Day 3
- Implemented `Policy Iteration` from Chapter 4 of the Sutton book to find the optimal policy for the gridworld environment.
- ![optimal policy](/day_003/optimal_policy.png)
- `day_003/policy_iteration.py` - Implementation of the policy evaluation with state-value function and greedy policy improvement
- `day_003/environment.py` and `day_003/gridworld.py` were shamelessly copied from the reference code base above.

--- 
# Day 4
- Implemented `Value Iteration` from Chapter 4 of the Sutton book to find the optimal policy for a 5 by 5 gridworld environment
- ![optimal policy](/day_004/optimal_policy.png)
- `day_004/value_iteration.py` - Implementation for the value iteration to find the optimal policy

---
# Day 5
- Started to implement Chapter 6, section 4 from the Sutton Book : SARSA On-Policy TD Control
- Couldn't get it to work, but I'll debug and try to fix tomorrow
- `day_005/SARSA.py` - Current implementation for SARSA

---
# Day 6
- Implemented SARSA and Q-learning and compared to a random agent baseline
- ![avg rewards](/day_006/avg_rewards.png)
- `day_006/agents.py` - Implementation of SARSA, QLearning and Random agents.

---
# Day 7 - 
- Started reading the [Atari paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) and I plan to implement this paper over the next few days.
- No coding / implementation for today
- [Day 8] Implemented the Policy Network for DQN

---
# Day 