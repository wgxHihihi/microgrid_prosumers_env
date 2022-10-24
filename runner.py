import pandas as pd
from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.log_every = 500
        self.train_every = 8

        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if not os.path.exists(self.args.save_dir + '/record/'):
            os.makedirs(self.args.save_dir + '/record/')

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        rewards = []
        # d = {'total reward': rewards[-1][0]}
        pbar = tqdm(range(self.args.max_episode_len))
        step = 0
        for ep in pbar:
            record = []
            self.env.day_index = 92
            s = self.env.reset()
            date = ''.join(self.env.date[self.env.day_index * 96].split(' ')[0].split('/'))

            reward = 0
            rr = np.zeros(self.env.n_agents)
            bess_revenue = 0
            for t in range(self.args.time_steps):
                u = []
                actions = []
                ori_u = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action, ori_a = agent.select_action(s[agent_id], self.noise, self.epsilon)
                        # print('s%d' % agent_id, s[agent_id])
                        u.append(action)
                        ori_u.append(ori_a)
                        actions.append(action)
                    if t == 0:
                        print(ori_u)
                        print(np.hstack(ori_u))
                # print(actions)
                s_next, r, done, info = self.env.step(actions)
                # print(s_next)
                r = [round(i, 2) for i in r]
                reward += sum(r)
                rr += r
                bess_revenue += info

                # log_data
                if (ep + 1) % self.log_every == 0:
                    building_state = np.stack(s[:self.env.building_num])
                    # print(building_state)
                    temp = building_state[:, 0]
                    soc = building_state[:, 1]
                    price = building_state[0, 5:7]
                    # print(s[self.env.building_num])
                    p_g = [s[self.env.building_num][0]]
                    # print(p_g)
                    bess_soc = [s[-1][0]]
                    rtp = [self.env.tou_buy[t + self.env.time_bias]]
                    outdoor_temp = [building_state[0, 4]]
                    action_pre = self.env.action_pre
                    p_demand_generated = [self.env.p_demand, self.env.p_generated]
                    p_buildings = building_state[:, 3]
                    # print(p_buildings)
                    record.append(np.hstack(
                        (
                            self.env.day_index, outdoor_temp, temp, p_buildings, soc, p_g, bess_soc, price, rtp,
                            np.hstack(actions),
                            action_pre,
                            p_demand_generated)))

                done_mask = [not d for d in done]
                self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents],
                                          s_next[:self.args.n_agents], done_mask)
                s = s_next
                step += 1
                # print(step)
                if self.buffer.current_size >= self.args.batch_size and ep >= 20 and step % self.train_every == 0:
                    step = 0
                    # print('train')
                    transitions = self.buffer.sample(self.args.batch_size)
                    for agent in self.agents:
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        agent.learn(transitions, other_agents)
                    # self.noise = max(0.05, self.noise - 0.0000005)
                    # self.epsilon = max(0.05, self.epsilon - 0.0000005)
                    self.noise = max(0.05, self.noise - 0.0000005)
                    self.epsilon = max(0.05, self.epsilon - 0.0000005)
                # np.save(self.save_path + '/returns.pkl', returns)

            # print(rr)
            rewards.append(np.hstack((reward, rr, bess_revenue)))
            pbar.set_description("Total reward: %.3f, Day: %d " % (reward, self.env.day_index))
            # print('ep: {}, total reward: {}'.format(ep, reward))
            if (ep + 1) % self.log_every == 0:
                pd.DataFrame(rewards, columns=['total',
                                               'building1', 'building2', 'building3', 'building4', 'building5',
                                               'dg', 'bess', 'bess_without_pen'
                                               ]).to_csv(self.args.save_dir + '/record/' + 'r_ep{}.csv'.format(ep + 1))
                pd.DataFrame(record,
                             columns=['day',
                                      'outdoor_temp',
                                      'temp1', 'temp2', 'temp3', 'temp4', 'temp5',
                                      'power_1', 'power_2', 'power_3', 'power_4', 'power_5',
                                      'soc1', 'soc2', 'soc3', 'soc4', 'soc5',
                                      'dg_p',
                                      'bess_soc',
                                      'buy', 'sold', 'rtp',
                                      'a1_ac', 'a1_ess', 'a2_ac', 'a2_ess', 'a3_ac', 'a3_ess',
                                      'a4_ac', 'a4_ess', 'a5_ac', 'a5_ess',
                                      'a_dg', 'a_bess',
                                      'a_true_1_ac', 'a_true_1_ess', 'a_true_2_ac', 'a_true_2_ess', 'a_true_3_ac',
                                      'a_true_3_ess', 'a_true_4_ac', 'a_true_4_ess', 'a_true_5_ac', 'a_true_5_ess',
                                      'a_true_dg', 'a_true_bess',
                                      'p_demand', 'p_generated'
                                      ]
                             ).to_csv(self.args.save_dir + '/record/' + 'record_{}_{}.csv'.format(ep + 1, date))

                for agent in self.agents:
                    agent.policy.save_model(ep)

        # def evaluate(self):
        #     returns = []
        #     for episode in range(self.args.evaluate_episodes):
        #         # reset the environment
        #         s = self.env.reset()
        #         rewards = 0
        #         for time_step in range(self.args.evaluate_episode_len):
        #             self.env.render()
        #             actions = []
        #             with torch.no_grad():
        #                 for agent_id, agent in enumerate(self.agents):
        #                     action = agent.select_action(s[agent_id], 0, 0)
        #                     actions.append(action)
        #             for i in range(self.args.n_agents, self.args.n_players):
        #                 actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
        #             s_next, r, done, info = self.env.step(actions)
        #             rewards += r[0]
        #             s = s_next
        #         returns.append(rewards)
        #         print('Returns is', rewards)
        #     return sum(returns) / self.args.evaluate_episodes
