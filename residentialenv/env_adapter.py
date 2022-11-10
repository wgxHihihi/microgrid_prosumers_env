from residentialenv.microgrid import microgrid
import numpy as np


class env_adapter:
    def __init__(self, seed):
        self.env = microgrid(seed)
        self.n_agents = self.env.n_agents
        # self.obs_shape = [5] * self.n_agents
        self.obs_shape = self.env.obs_space
        self.state_shape = sum(self.env.obs_space)
        self.action_shape = self.env.act_space
        self.n_players = self.env.n_agents
        self.n_actions = 11

    def close(self):
        pass

    def get_obs(self):
        # print(np.vstack(self.env.state))
        return np.vstack(self.env.state)

    def get_state(self):
        return np.hstack(self.env.state)

    def get_avail_agent_actions(self, agent_id):
        return [1] * self.n_actions

    def action_adapter(self, actions):
        new_actions = np.array([[-1 + 2 / self.n_actions * a] for a in actions])

        return new_actions

    @staticmethod
    def reward_refactor(rewards: list, info: dict, n_agents: int):
        power_limit_penalty = info['power_limit_penalty']
        net_cost = info['net_cost']
        rewards_new = [r * 1 for r in rewards]
        rewards_new = [sum(rewards_new) + power_limit_penalty + net_cost] * n_agents
        return rewards_new

    def step(self, actions):
        # print(actions)
        state, state_share, r, done, info, _ = self.env.step(actions)
        rewards = self.reward_refactor(r, info, self.n_agents)
        # state = self.limit_state(state)
        return state, rewards, done, info, r

    def reset(self):
        self.env.day_index = 1
        state, state_shared, info = self.env.reset()
        # state = self.limit_state(state)
        return state, info

    def limit_state(self, state):
        return [s[:5] for s in state]
