# SARSA should inherit from some agent
class SARSA():
    def observe(self, exp: Experience):
        obs_old, act_old, reward, obs_new, act_new = exp.Experience

        self.Q[obs_old, act_old] += self.lr * (reward + (self.gamma * self.Q[obs_new, act_new]) - self.Q[obs_old, act_old])

    def run_episode(self, seed) -> List[int]:
        obs
        act = self.get_action(obs)
        terminated = False
        rewards = []
        
        while not terminated:
            (obs_new, reward, truncated, terminatd, info) = self.env.step(act)
            act_new = self.get_action(obs_new)
            exp = Experience(obs, act, reward, obs_new, act_new)
            self.observe(exp)
            rewards.append(reward)
            obs = obs_new
            act = act_new

        return reward