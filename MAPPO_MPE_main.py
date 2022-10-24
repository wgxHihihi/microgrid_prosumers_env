import random

import pandas as pd
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo import MAPPO
# from make_env import make_env
from residentialenv.microgrid import microgrid
import math
import os
from tqdm import tqdm


class Runner_MAPPO:
    def __init__(self, args, seed):
        self.args = args
        self.seed = seed
        self.log_every = 100
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = microgrid(self.seed)
        self.args.N = self.env.n_agents  # The number of agents
        self.args.obs_dim_n = self.env.obs_space  # obs dimensions of N agents
        self.args.action_dim_n = self.env.act_space  # actions dimensions of N agents
        self.args.episode_limit = 96

        # Create N agents
        self.agents = self._init_agents()
        self.agent_num = self.env.n_agents

        print("observation_space=", self.env.obs_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(
            log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format('microgrid', self.env.n_agents, self.seed))

        self.log_path = './train_logs/mappo/result_%d/' % self.seed
        # record path
        if not os.path.exists(self.log_path + 'record/'):
            os.makedirs(self.log_path + 'record/')
            print('------make dir: \'{}\'------'.format(self.log_path + 'record/'))
        if not os.path.exists(self.log_path + 'model/'):
            os.makedirs(self.log_path + 'model/')
            print('------make dir: \'{}\'------'.format(self.log_path + 'model/'))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def _init_agents(self):
        agents = []
        # print(self.args)
        for i in range(self.args.N):
            agent = MAPPO(self.args, i)
            agents.append(agent)
        return agents

    def run(self):
        rewards = []
        is_record = False
        for ep in tqdm(range(self.args.max_train_steps)):
            buffer_rewards = []
            buffer_rewards_dis = np.zeros(self.agent_num + 2)
            total_steps = 0
            if (ep + 1) % self.log_every == 0:
                is_record = True
            while self.replay_buffer.episode_num < self.args.batch_size:
                ep_r, ep_steps, ep_r_dis, record, date = self.run_episode(is_record=is_record)
                if is_record:
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
                                 ).to_csv(
                        self.log_path + 'record/' + 'record_{}_{}.csv'.format(ep + 1, date))
                    is_record = False
                buffer_rewards.append(ep_r)
                buffer_rewards_dis += ep_r_dis
                total_steps += ep_steps

            print('train_ep: {}, collection completed, mean reward = {}'
                  .format(ep, sum(buffer_rewards) / len(buffer_rewards)))
            # print(buffer_rewards_dis)
            rewards.append(np.hstack(
                ([sum(buffer_rewards) / len(buffer_rewards)], buffer_rewards_dis / len(buffer_rewards))))
            # print(rewards)
            for i in range(self.agent_num):
                self.agents[i].train(self.replay_buffer, ep)

            if (ep + 1) % self.log_every == 0:
                pd.DataFrame(rewards,
                             columns=['total',
                                      'home1', 'home2', 'home3', 'home4', 'home5',
                                      'power_limit_penalty', 'net_cost']
                             ).to_csv(self.log_path + 'record/' + 'r_{}.csv'.format(ep + 1))
                for agent in self.agents:
                    agent.save_model(self.log_path + 'model/' + '{}_agent_{}_actor.pth'.format(ep + 1, agent.agent_id),
                                     self.log_path + 'model/' + '{}_agent_{}_critic.pth'.format(ep + 1, agent.agent_id))
            self.replay_buffer.reset_buffer()

    def reward_refactor(self, rewards: list, info: dict):
        power_limit_penalty = info['power_limit_penalty']
        net_cost = info['net_cost']
        rewards_new = [r * 1 for r in rewards]
        rewards_new = [sum(rewards_new) + power_limit_penalty + net_cost] * self.env.n_agents
        return rewards_new

    def run_episode(self, evaluate=False, is_record=False):
        record = []
        episode_reward = 0
        ep_reward_dis = np.zeros(self.agent_num + 2)
        self.env.day_index = 1
        obs_n, _, info = self.env.reset()
        if is_record:
            date = ''.join(self.env.date[self.env.day_index * 96].split(' ')[0].split('/'))
        else:
            date = None

        # print('day: ', self.env.day_index)
        # print(obs_n)
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        for episode_step in range(self.args.episode_limit):
            # Get actions and the corresponding log probabilities of N agents
            actions = []
            a_logprobs = []
            values = []
            # print(obs_n)

            s = np.hstack(obs_n).flatten()
            # print(obs_n)
            for i in range(self.agent_num):
                a_i, a_logprob_i = self.agents[i].choose_action(obs_n[i], evaluate=evaluate)
                v_n = self.agents[i].get_value(s)
                if (np.isnan(a_i[0])):
                    print('a[{}][{}] is nan!'.format(i, 0))
                if (np.isnan(all(obs_n[i]))):
                    print('obs[{}] has nan!'.format(i))
                # if i > 0:
                #     actions.append(np.array([0]))
                # else:
                actions.append(a_i)
                a_logprobs.append(a_logprob_i.sum())
                values.append(v_n)
            # print(actions)
            if is_record:
                building_state = np.stack(obs_n[:self.env.building_num])
                # print(building_state)
                # temp = building_state[:, 0]
                soc = building_state[:, 0]
                price = [self.env.local_buy, self.env.local_sold, self.env.net_buy, self.env.net_sell]
                # print(s[self.env.building_num])
                # p_g = [obs_n[self.env.building_num][0]]
                # # print(p_g)
                # bess_soc = [obs_n[-1][0]]
                rtp = [self.env.tou_buy[episode_step + self.env.time_bias]]
                outdoor_temp = [building_state[0, 3]]
                action_pre = self.env.action_pre
                p_demand_generated = [self.env.p_demand, self.env.p_generated, self.env.p_demand - self.env.p_generated]
                p_buildings = building_state[:, 2]
                p_fixed = building_state[:, 1]
                net_cost = [info['net_cost']]
                # print(p_buildings)
                record.append(np.hstack(
                    (
                        self.env.time, outdoor_temp, p_buildings, p_fixed, soc, price, rtp,
                        np.hstack(actions), action_pre, p_demand_generated, net_cost)))
                # print(record)

            obs_next_n, _, r_n, done_n, info, _ = self.env.step(actions)
            episode_reward += sum(r_n) + sum(info.values())
            ep_reward_dis += r_n + list(info.values())
            # print(r_n)
            # r_n = [sum(r_n)] * self.env.n_agents
            r_n = self.reward_refactor(r_n, info)
            # print(r_n)
            if self.args.use_reward_norm:
                r_n = self.reward_norm(r_n)
            elif self.args.use_reward_scaling:
                r_n = self.reward_scaling(r_n)

            # Store the transition
            # print(a_logprobs)
            # print(r_n)
            self.replay_buffer.store_transition(episode_step, obs_n, s, values, actions, a_logprobs, r_n, done_n)

            obs_n = obs_next_n
            if all(done_n):
                break

        if not evaluate:
            # An episode is over, store v_n in the last step
            # s = np.hstack(obs_n).flatten()
            v = []
            for i in range(self.agent_num):
                v_n = self.agents[i].get_value(s)
                v.append(v_n)
            self.replay_buffer.store_last_value(episode_step + 1, v)

        return episode_reward, episode_step + 1, ep_reward_dis, record, date


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(5000), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=96, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=256,
                        help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--hidden_layers", type=int, default=2, help="The number of hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True,
                        help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=False, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False,
                        help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")

    args = parser.parse_args()
    for i in range(2):
        seed = random.randint(2, 20)
        runner = Runner_MAPPO(args, seed=seed)
        runner.run()
