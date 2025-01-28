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
