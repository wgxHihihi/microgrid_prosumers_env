import numpy as np


class BESS:
    def __init__(self, bess_args, delta_t):
        self.e_max = bess_args.e_max
        self.p_max = bess_args.p_max
        self.soc_max = bess_args.soc_max
        self.soc_min = bess_args.soc_min
        self.deg_cost = bess_args.deg_cost
        self.soc = 0.2
        self.delta_soc = 0
        self.state = []
        self.delta_t = delta_t
        self.a_bess = 0
        self.p = 0

    def reset(self):
        # self.soc = self.soc_min
        self.state = []
        self.p = 0
        self.state.append(self.soc)
        return self.state

    def soc_constraint(self, power):
        # 电池空了之后只允许充电
        if power >= 0:
            p_new = np.clip(power, 0,
                            min(self.p_max, ((self.soc_max - self.soc) * self.e_max / 0.95) / self.delta_t))
            # 放电不能过量
        else:
            p_new = np.clip(power,
                            max(-self.p_max, ((self.soc_min - self.soc) * self.e_max * 0.95) / self.delta_t), 0)
        return p_new

    def __SoC(self, soc_t, power):
        if power >= 0:
            # charge
            soc_t_1 = soc_t + 0.95 * power * self.delta_t / self.e_max
        else:
            # discharge
            soc_t_1 = soc_t + (1 / 0.95) * power * self.delta_t / self.e_max
        soc_t_1 = np.clip(soc_t_1, self.soc_min, self.soc_max)
        return soc_t_1

    def step(self, power):
        self.a_bess = power
        power = np.clip(power * self.p_max, -self.p_max, self.p_max)
        self.p = self.soc_constraint(power)
        soc_t = self.__SoC(self.soc, self.p)
        self.delta_soc = abs(soc_t - self.soc)
        self.soc = soc_t
        self.state = [self.soc]

        return self.soc

    def reward_fn(self, buy_price, sold_price):
        if self.p >= 0:
            price = buy_price
        else:
            price = sold_price
        ele_cost = price * self.p * self.delta_t

        # penalty about the illegal charging/discharging actions
        soc_penalty = 0
        if self.soc == self.soc_min and self.a_bess < 0:
            soc_penalty = -1
        elif self.soc == self.soc_max and self.a_bess > 0:
            soc_penalty = -1

        degradation_cost = self.delta_soc * self.deg_cost
        return - ele_cost - degradation_cost + soc_penalty, - ele_cost - degradation_cost
