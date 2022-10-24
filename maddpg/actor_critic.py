import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(args.obs_shape[agent_id], args.hiden_layer)
        self.fc2 = nn.Linear(args.hiden_layer, int(1 * args.hiden_layer))
        self.fc3 = nn.Linear(int(1 * args.hiden_layer), int(1 * args.hiden_layer))
        self.action_out = nn.Linear(int(1 * args.hiden_layer), args.action_shape[agent_id])

        self.bn1 = nn.BatchNorm1d(args.obs_shape[agent_id])

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.action_out)

    def forward(self, x, is_train=True):
        # print(x)
        # if is_train:
        #     # print(x[0])
        #     x = self.bn1(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        actions = torch.tanh(self.action_out(x))
        # actions = self.action_out(x)
        return actions


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(sum(args.obs_shape), args.hiden_layer)
        self.fc2 = nn.Linear(args.hiden_layer + sum(args.action_shape), args.hiden_layer)
        self.fc3 = nn.Linear(args.hiden_layer, args.hiden_layer)
        self.q_out = nn.Linear(args.hiden_layer, 1)
        self.bn1 = nn.BatchNorm1d(sum(args.obs_shape))
        self.bn2 = nn.BatchNorm1d(args.hiden_layer + sum(args.action_shape))

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc3)
        orthogonal_init(self.q_out)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        action = torch.cat(action, dim=1)
        # print(state)
        # state = self.bn1(state)
        x = torch.tanh(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        # x = self.bn2(x)
        x = torch.tanh(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
