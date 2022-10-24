import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim_n
        self.state_dim = sum(self.obs_dim)
        self.action_dim = args.action_dim_n
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()
        # create a buffer (dictionary)

    def reset_buffer(self):
        self.buffer = dict()
        self.buffer['s'] = np.empty([self.batch_size, self.episode_limit, self.state_dim])
        self.buffer['a_logprob_n'] = np.empty([self.batch_size, self.episode_limit, self.N])
        self.buffer['v_n'] = np.empty([self.batch_size, self.episode_limit + 1, self.N])
        self.buffer['done_n'] = np.empty([self.batch_size, self.episode_limit, self.N])
        self.buffer['r_n'] = np.empty([self.batch_size, self.episode_limit, self.N])
        for i in range(self.N):
            self.buffer['obs_%d' % i] = np.empty([self.batch_size, self.episode_limit, self.obs_dim[i]])
            self.buffer['a_%d' % i] = np.empty([self.batch_size, self.episode_limit, self.action_dim[i]])
        #
        # self.buffer = {'obs_n': np.empty([self.batch_size, self.episode_limit, self.N]),
        #                's': np.empty([self.batch_size, self.episode_limit, self.state_dim]),
        #                'v_n': np.empty([self.batch_size, self.episode_limit + 1, self.N]),
        #                'a_n': np.empty([self.batch_size, self.episode_limit, self.N]),
        #                'a_logprob_n': np.empty([self.batch_size, self.episode_limit, self.N]),
        #                'r_n': np.empty([self.batch_size, self.episode_limit, self.N]),
        #                'done_n': np.empty([self.batch_size, self.episode_limit, self.N])
        #                }
        self.episode_num = 0
        self.adv = []

    def store_transition(self, episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n):
        for i in range(self.N):
            self.buffer['obs_%d' % i][self.episode_num][episode_step] = obs_n[i]
            self.buffer['a_%d' % i][self.episode_num][episode_step] = a_n[i]
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n
        self.buffer['r_n'][self.episode_num][episode_step] = r_n
        self.buffer['done_n'][self.episode_num][episode_step] = done_n

    def store_last_value(self, episode_step, v_n):
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.episode_num += 1

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
            else:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch
