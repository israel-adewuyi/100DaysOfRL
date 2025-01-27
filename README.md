I'm working on / learning RL and sharing my progress publicly. 

Library - [Gymnasium](https://gymnasium.farama.org/)
Main reference code : [ARENA RL chapter](https://github.com/callummcdougall/ARENA_3.0/tree/main/chapter2_rl)

# Day 1
- Worked on the Multi-arm bandit problem
-- Implemented the epsilon-greedy action value method and compared different epsilon values. From the graph below, it seems they all converge given a long enough episode.
![reward averaging results](/day_001/artefacts/average_rewards.png)
- `day_001/multi_armed_bandit_env.py` - implementation of the multi armed bandit environment
- `day_001/agent.py` -Implementation of the epsilon-greedy agent and random agent
- `day_001/main.py` - code to run the agents
- `day_001/utils.py` and `day_001/main.py` were shamelessly copied from the reference code above, with some slight modifications.

---
