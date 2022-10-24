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
        self.hidden_layer = []
        for i in range(args.hiden_layer_num):
            self.hidden_layer.append(nn.Linear(args.hiden_layer, args.hiden_layer))
        self.action_out = nn.Linear(args.hiden_layer, args.action_shape[agent_id])

        orthogonal_init(self.fc1)
        for lay in self.hidden_layer:
            orthogonal_init(lay)
        orthogonal_init(self.action_out)

    def forward(self, x, is_train=True):
        x = torch.relu(self.fc1(x))
        for lay in self.hidden_layer:
            x = torch.relu(lay(x))
        actions = torch.tanh(self.action_out(x))
        return actions


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(sum(args.obs_shape), args.hiden_layer)
        self.fc2 = nn.Linear(args.hiden_layer + sum(args.action_shape), args.hiden_layer)
        self.hidden_layer = []
        for i in range(args.hiden_layer_num - 1):
            self.hidden_layer.append(nn.Linear(args.hiden_layer, args.hiden_layer))
        self.q_out = nn.Linear(args.hiden_layer, 1)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        for lay in self.hidden_layer:
            orthogonal_init(lay)
        orthogonal_init(self.q_out)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        action = torch.cat(action, dim=1)
        x = torch.tanh(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = torch.tanh(self.fc2(x))
        for lay in self.hidden_layer:
            x = torch.tanh(lay(x))
        q_value = self.q_out(x)
        return q_value
