import numpy as np


class DG:
    def __init__(self, DG_args, delta_t):
        """
        状态：输出功率
        动作：u in [0,1]
        """
        self.p_max = DG_args.p_max
        self.T_dg = DG_args.T_dg
        self.delta_t = delta_t
        self.lam_1 = DG_args.lam_1
        self.lam_2 = DG_args.lam_2
        self.power = 0
        self.state = np.zeros(1)
        self.action_pre = 0

    def reset(self):
        self.power = 0.0
        self.state = np.array([self.power])
        return self.state

    def step(self, u):
        u = np.clip(u, -1, 1)
        u = (u + 1) / 2.0
        self.action_pre = u
        self.power = (1 - self.delta_t / self.T_dg) * self.power + (self.p_max * self.delta_t) * u / self.T_dg
        self.power = max(self.power, 0)
        self.state[0] = self.power
        # print(self.power)
        return self.state

    def reward_fn(self, price):
        reward = price * self.power - (self.lam_1 * pow(self.power, 2) + self.lam_2 * self.power)
        return reward * self.delta_t
