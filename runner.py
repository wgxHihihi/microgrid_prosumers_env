from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.args.save_dir + '/record/'):
            os.makedirs(self.args.save_dir + '/record/')
        self.log_every = 100
        self.train_every = 8

    def _init_agents(self):
        # agents = []
        # for i in range(self.args.n_agents):
        agent = Agent(self.args.n_agents, self.args)
        # agents.append(agent)
        return agent

    def run(self):
        rewards = []
        for ep in tqdm(range(self.args.episode)):
            # reset the environment
            record = []

            is_record = (ep + 1) % self.log_every == 0
            s, info = self.env.reset()
            ep_reward = 0
            rewards_dis = np.zeros(self.env.n_agents + 2)
            for t in range(self.args.max_episode_len):
                u = []
                actions = []
                with torch.no_grad():
                    for agent_id in range(self.args.n_agents):
                        action = self.agents.select_action(s[agent_id], self.noise, self.epsilon)
                        u.append(action)
                        actions.append(action)

                    # log_data
                    if is_record:
                        building_state = np.stack(s[:self.env.env.building_num])
                        soc = building_state[:, 0]
                        price = [self.env.env.local_buy, self.env.env.local_sold, self.env.env.net_buy,
                                 self.env.env.net_sell]
                        rtp = [self.env.env.tou_buy[t + self.env.env.time_bias]]
                        outdoor_temp = [building_state[0, 3]]
                        action_pre = self.env.env.action_pre
                        p_demand_generated = [self.env.env.p_demand, self.env.env.p_generated,
                                              self.env.env.p_demand - self.env.env.p_generated]
                        p_buildings = building_state[:, 2]
                        p_fixed = building_state[:, 1]
                        net_cost = [info['net_cost']]
                        record.append(np.hstack(
                            (self.env.env.time, outdoor_temp, p_buildings, p_fixed, soc, price, rtp,
                             np.hstack(actions), action_pre, p_demand_generated, net_cost)))
                s_next, r, done, info, r_dis = self.env.step(actions)
                self.buffer.store_episode(s, u, r, s_next)
                s = s_next
                ep_reward += r[0]
                rewards_dis += r_dis + list(info.values())

                if self.buffer.current_size >= self.args.batch_size and t % self.train_every == 0:
                    transitions = self.buffer.sample(self.args.batch_size)
                    self.agents.learn(transitions)
                    self.noise = max(0.05, self.noise - 0.0000005)
                    self.epsilon = max(0.05, self.epsilon - 0.0000005)

                if done[0]:
                    print("ep:{}, ep_reward:{}".format(ep, ep_reward))
                    break

            rewards.append(np.hstack((ep_reward, rewards_dis)))
            if is_record:
                pd.DataFrame(rewards,
                             columns=['total',
                                      'home1', 'home2', 'home3', 'home4', 'home5',
                                      'power_limit_penalty', 'net_cost'
                                      ]
                             ).to_csv(self.args.save_dir + '/record/' + 'r_ep{}.csv'.format(ep + 1))
                pd.DataFrame(record,
                             columns=['time',
                                      'outdoor_temp',
                                      'power_1', 'power_2', 'power_3', 'power_4', 'power_5',
                                      'power_fix_1', 'power_fix_2', 'power_fix_3', 'power_fix_4', 'power_fix_5',
                                      'soc1', 'soc2', 'soc3', 'soc4', 'soc5',
                                      'local_buy', 'local_sold', 'net_buy', 'net_sold', 'rtp',
                                      'a1_ess', 'a2_ess', 'a3_ess', 'a4_ess', 'a5_ess',
                                      'a_true_1_ess', 'a_true_2_ess', 'a_true_3_ess', 'a_true_4_ess',
                                      'a_true_5_ess',
                                      'p_demand', 'p_generated', 'p_net',
                                      'net_cost'
                                      ]
                             ).to_csv(self.args.save_dir + '/record/' + 'record_{}.csv'.format(ep + 1))

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                # self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
