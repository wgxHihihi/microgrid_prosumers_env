# from residentialenv.appliances import *
import numpy as np
import random
import pandas as pd

timeslots = 96


class building_env:
    """
    用户环境：ESS，随机负荷
    状态：ev_soc，固定负荷，总负荷，室外温度
    动作：储能功率
    奖励：
    """

    def __init__(self, project_dir, building_args, seed, pass_state_len, start_index):
        random.seed(seed)
        self.delta_t = 24 / timeslots
        self.time_bias = pass_state_len
        self.index = 0
        # ESS

        self.soc = building_args.soc_ini
        self.ess_p_max = building_args.ess_p_max
        self.soc_max = building_args.soc_max
        self.soc_min = building_args.soc_min
        self.ess_e_max = building_args.ess_e_max
        self.a_ess = 0

        # Nonshiftable Apps
        self.fixed_Power, self.time_array = self.data_loader(project_dir + building_args.data_path)  # 随机的固定负荷
        self.time = str()
        # 总功率
        self.p_total = 0

        # 状态
        self.state = np.array([self.soc, self.fixed_Power[0], 0, 0, 0])

        self.action_pre = np.zeros(1)

        self.pass_state = self.fixed_Power[start_index - self.time_bias:start_index]

    def reset(self, T_out, start_index):
        self.p_total = self.fixed_Power[start_index]
        self.time = self.time_array[start_index]
        # self.a_ess = 0
        state_curr = np.hstack(
            [self.soc, np.array([self.fixed_Power[start_index], self.p_total, T_out])])

        self.pass_state = self.fixed_Power[start_index - self.time_bias:start_index]
        self.state = np.hstack([state_curr, self.pass_state])
        # print(self.time,self.p_total)
        return self.state

    @staticmethod
    def data_loader(path):
        fixed_power_data = pd.read_csv(path)
        return np.array(fixed_power_data['grid']), np.array(fixed_power_data['local_15min'])

    def soc_constraint(self, power):
        # 电池空了之后只允许充电
        if power >= 0:
            p_new = np.clip(power, 0,
                            min(self.ess_p_max, ((self.soc_max - self.soc) * self.ess_e_max / 0.95) / self.delta_t))
        else:
            p_new = np.clip(power,
                            max(-self.ess_p_max, ((self.soc_min - self.soc) * self.ess_e_max * 0.95) / self.delta_t), 0)
        return p_new

    def a_constraint(self, actions):
        a = actions.copy()
        a = self.ess_p_max * a
        a = self.soc_constraint(a)
        return a

    def __SoC(self, soc_t, power):
        if power >= 0:
            soc_t_1 = soc_t + 0.95 * power * self.delta_t / self.ess_e_max
        else:
            soc_t_1 = soc_t + (1 / 0.95) * power * self.delta_t / self.ess_e_max
        soc_t_1 = np.clip(soc_t_1, 0, self.soc_max)
        return soc_t_1

    def step(self, action, start_index, time, T_out):
        """
        :param action:AC,ESS
        :param start_index: start index
        :param time: time slot
        :param T_out: outdoor temperature
        :return: next state
        """
        self.a_ess = action
        action_new = self.a_constraint(action)
        self.action_pre = action_new
        self.p_total = 0
        self.p_total += sum(action_new)
        self.p_total += self.fixed_Power[start_index + time]
        self.time = self.time_array[start_index + time]

        self.soc = self.__SoC(self.soc, action_new)
        # print(self.soc)
        state_curr = np.hstack([self.soc,
                                np.array([self.fixed_Power[start_index + time], self.p_total, T_out])])
        """
        状态：室内温度，储能电量，固定负荷，室外温度，时间
        """
        self.pass_state = np.append(self.pass_state[1:], self.fixed_Power[start_index + time - 1])
        # print(self.pass_state)
        self.state = np.hstack([state_curr, self.pass_state])
        # next_state
        # self.state = np.array([self.T_in, self.soc, self.fixed_Power[start_index + time], T_out, self.p_total])
        # done = True if self.T_in <= 20 or self.T_in >= 28 else False

        # reward, ele_cost, therm_comfort = self.reward_fn(price_t)

        # done = True if time == 95 else False

        return self.state  # , done  # , reward, done, ele_cost, therm_comfort

    def reward_fn(self, buy_price, sold_price, time):
        # electricity cost
        if self.p_total >= 0:
            price = buy_price
        else:
            price = sold_price
        e_cost = self.p_total * self.delta_t * price
        r = - e_cost

        return r, e_cost
