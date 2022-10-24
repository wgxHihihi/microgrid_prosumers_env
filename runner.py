import pandas as pd
from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.env_info_refactor import reward_refactor


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
        self.log_every = 100
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
        print('hidden_layer_num: %d' % self.args.hiden_layer_num)
        print('hidden_layer_dim: %d' % self.args.hiden_layer)
        return agents

    def run(self):
        rewards = []
        # d = {'total reward': rewards[-1][0]}
        pbar = tqdm(range(self.args.max_episode_len))
        step = 0
        for ep in pbar:
            record = []
            self.env.day_index = 1
            s, con_s, info = self.env.reset()
            date = ''.join(self.env.date[self.env.day_index * 96].split(' ')[0].split('/'))

            episode_reward = 0
            rewards_dis = np.zeros(self.env.n_agents + 2)
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

                # log_data
                if (ep + 1) % self.log_every == 0:
                    building_state = np.stack(s[:self.env.building_num])
                    soc = building_state[:, 0]
                    price = [self.env.local_buy, self.env.local_sold, self.env.net_buy, self.env.net_sell]
                    rtp = [self.env.tou_buy[t + self.env.time_bias]]
                    outdoor_temp = [building_state[0, 3]]
                    action_pre = self.env.action_pre
                    p_demand_generated = [self.env.p_demand, self.env.p_generated,
                                          self.env.p_demand - self.env.p_generated]
                    p_buildings = building_state[:, 2]
                    p_fixed = building_state[:, 1]
                    net_cost = [info['net_cost']]
                    record.append(np.hstack(
                        (self.env.time, outdoor_temp, p_buildings, p_fixed, soc, price, rtp,
                         np.hstack(actions), action_pre, p_demand_generated, net_cost)))

                s_next, _, r, done, info, _ = self.env.step(actions)
                episode_reward += sum(r) + sum(info.values())
                rewards_dis += r + list(info.values())
                r = reward_refactor(r, info, self.env.n_agents)

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

            # print(rr)
            rewards.append(np.hstack((episode_reward, rewards_dis)))
            pbar.set_description("Total reward: %.3f, Day: %d " % (episode_reward, self.env.day_index))
            # print('ep: {}, total reward: {}'.format(ep, reward))
            if (ep + 1) % self.log_every == 0:
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
                             ).to_csv(self.args.save_dir + '/record/' + 'record_{}_{}.csv'.format(ep + 1, date))

                for agent in self.agents:
                    agent.policy.save_model(ep)
