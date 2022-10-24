import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG
import copy

class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, noise_rate, epsilon, is_random=True):
        if np.random.uniform() < epsilon and is_random:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
            ori_u = 2
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_network(inputs, is_train=False).squeeze(0)
            ori_u = copy.deepcopy(pi)
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(
                self.args.action_shape[self.agent_id])  # gaussian noise
            # ori_u = u
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u, ori_u

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)
