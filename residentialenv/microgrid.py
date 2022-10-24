import os

import pandas as pd
from residentialenv.single_home_env import building_env
from residentialenv.config.config_loader import *
import numpy as np
from gym.spaces.box import Box


class microgrid:
    def __init__(self, seed):
        """
        buildings: 5
        obs_space: [buildings]
        act_space: [buildings]
        price: time-of-use scheme
        local market rule: 1. all the elements can sell/buy energy in the local market.
                           2. local prices are adjusted based on the difference between demand and generated power.
                           3. selling/buying too much to/from the utility grid will suffered from a penalty price.
        """
        self.time_bias = 0
        self.pass_state_len = 16
        self.seed = seed
        self.day_index = 1
        self.clock = 0

        self.delta_t = 0.25
        self.obs_space = [9 + self.pass_state_len] * 5
        self.observation_space = [Box(shape=(dim,), low=float('-inf'), high=float('inf')) for dim in self.obs_space]
        self.share_observation_space = [Box(shape=(sum(self.obs_space),), low=float('-inf'),
                                            high=float('inf'))] * len(self.observation_space)
        self.act_space = [1] * 5
        self.action_space = [Box(shape=(dim,), low=-1, high=1) for dim in self.act_space]
        self.n_agents = len(self.observation_space)

        self.buildings_args, self.project_dir = self.__load_configs()

        self.rtp_path = self.project_dir + '/residentialenv/training_data/price.csv'
        self.Tout_path = self.project_dir + '/residentialenv/training_data/temp.csv'
        """
        buildings
        状态：储能电量，固定负荷，总负荷，室外温度，时间，买电价格（上一时刻的），售电价格（上一时刻的）,电力需求（上一时刻的）,发电量（上一时刻的）；
        动作：储能功率；
        奖励函数：-买电成本
        """
        self.building_num = len(self.buildings_args)
        self.buildings = []
        for args in self.buildings_args:
            self.buildings.append(
                building_env(self.project_dir, args, self.seed, self.pass_state_len,
                             self.day_index * 96 + self.time_bias))

        # outdoor temperature

        self.tempdata = pd.read_csv(self.Tout_path)
        self.T_out = self.tempdata['T_out']
        self.date = self.tempdata['time']

        # time-of-use price
        self.tou_buy = pd.read_csv(self.rtp_path)['time-of-use']
        self.tou_sold = self.tou_buy * 0.8

        # local price (last time)
        self.local_buy = self.tou_buy[0 + self.time_bias]
        self.local_sold = 0.9 * self.local_buy  # 本地卖电价格是本地买点价格的0.9，10%的损耗

        # demand/generated power
        self.p_demand = 0
        self.p_generated = 0

        # power limitation
        self.p_limit = 12
        self.ibr_rate = 0.44

        self.action_pre = []
        self.state = []
        self.grid_state = []

        self.time = str()

        self.net_buy = self.tou_buy[0 + self.time_bias]
        self.net_sell = self.tou_sold[0 + self.time_bias]

    @staticmethod
    def __load_configs():
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path_buildings = project_dir + '/residentialenv/config/buildings_config.yaml'
        buildingargs = load_config(path_buildings)
        return buildingargs, project_dir

    def cal_local_price(self, rtp_buy, rtp_sold):
        local_buy_temp = rtp_buy
        local_sold_temp = 0.9 * local_buy_temp
        local_buy = 0
        local_sold = 0
        net_buy = rtp_buy
        net_sell = rtp_sold
        p_g = self.p_demand - self.p_generated

        if p_g == 0:
            local_buy = local_buy_temp
            local_sold = local_sold_temp
        elif p_g < 0:
            # p_generated>p_demand 向电网卖电
            local_buy = local_buy_temp
            sold_power = abs(p_g)
            if sold_power > self.p_limit:
                # penalty selling price
                net_sell = (1 - self.ibr_rate) * rtp_sold
            else:
                net_sell = rtp_sold
            local_sold = (net_sell * sold_power + local_sold_temp * self.p_demand) / self.p_generated
        elif p_g > 0:
            # p_generated<p_demand 买电
            local_sold = local_sold_temp
            buy_power = abs(p_g)
            if buy_power > self.p_limit:
                # penalty buying price
                net_buy = (1 + self.ibr_rate) * rtp_buy
            else:
                net_buy = rtp_buy
            local_buy = (net_buy * buy_power + local_buy_temp * self.p_generated) / self.p_demand

        return local_buy, local_sold, net_buy, net_sell

    def reset(self):
        self.state = []
        self.p_demand = 0
        self.p_generated = 0
        self.action_pre = [0] * sum(self.act_space)
        self.clock = 0
        start_index = self.day_index * 96 + self.time_bias
        # print(start_index)
        # reset buildings
        for building in self.buildings:
            building.reset(self.T_out[start_index], start_index)
            if building.p_total > 0:
                self.p_demand += building.p_total
            else:
                self.p_generated += abs(building.p_total)
        self.time = self.buildings[0].time
        # calculate local prices
        self.local_buy, self.local_sold, self.net_buy, self.net_sell = self.cal_local_price(
            self.tou_buy[0 + self.time_bias],
            self.tou_sold[0 + self.time_bias])

        # get buildings' state
        self.grid_state = np.array(
            [self.local_buy, self.local_sold, self.p_demand, self.p_generated, self.clock])

        for building in self.buildings:
            state = building.state
            state = np.hstack((state, self.grid_state))
            self.state.append(state)

        r_power_limit, net_cost = self.reward_fn()
        info = {'power_limit_penalty': r_power_limit, 'net_cost': net_cost}
        # print(self.time)
        return self.state, np.concatenate(self.state), info

    def step(self, action):
        a = action.copy()
        # print(self.time)
        a_homes = a[:self.building_num]
        start_index = self.day_index * 96 + self.time_bias

        # reset the demand/generated power
        self.p_demand = 0
        self.p_generated = 0

        self.action_pre = []
        self.clock += 1
        # step the buildings
        for i, building in enumerate(self.buildings):
            building.step(a_homes[i], start_index, self.clock, self.T_out[start_index + self.clock])
            self.action_pre += building.action_pre.tolist()

            if building.p_total > 0:
                self.p_demand += building.p_total
            else:
                self.p_generated += abs(building.p_total)

        # calculate the local prices
        self.local_buy, self.local_sold, self.net_buy, self.net_sell = self.cal_local_price(
            self.tou_buy[self.clock + self.time_bias],
            self.tou_sold[self.clock + self.time_bias])
        """
        Note: The local_buy and local_sold prices at present time slot are unknown when all the agents making decision.
              The local_buy and local_sold in the observations of agent belong to the last time slot.
              Therefore, there is a game (Static Game) between all agents!
        """
        self.grid_state = np.array([self.local_buy, self.local_sold, self.p_demand, self.p_generated, self.clock])
        rewards = []
        state_next = []

        # get buildings' states and rewards
        for building in self.buildings:
            state_next.append(np.hstack((building.state, self.grid_state)))
            r, ele_cost = building.reward_fn(self.local_buy, self.local_sold, self.clock)
            rewards.append(r)

        r_power_limit, net_cost = self.reward_fn()

        done = [(True if self.clock == 96 else False)] * self.n_agents
        if done[0]:
            self.clock = 0
        self.state = state_next
        self.time = self.buildings[0].time
        info = {'power_limit_penalty': r_power_limit, 'net_cost': net_cost}
        return self.state, np.concatenate(self.state), rewards, done, info, None

    def reward_fn(self):
        p_g = abs(self.p_demand - self.p_generated)
        r_power_limit = -0.1 * max(p_g - self.p_limit, 0)
        net_cost = p_g * self.delta_t * (-self.net_buy if self.p_demand > self.p_generated else self.net_sell)
        return r_power_limit, net_cost
