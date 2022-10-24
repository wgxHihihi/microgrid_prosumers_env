import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import *


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor_RNN(nn.Module):
    def __init__(self, args, agent_id, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim_n[agent_id])
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        prob = torch.softmax(self.fc2(self.rnn_hidden), dim=-1)
        return prob


class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


class Actor_MLP(nn.Module):
    def __init__(self, args, agent_id, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc_hidden = []
        for i in range(args.hidden_layers):
            self.fc_hidden.append(nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim))
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim_n[agent_id])
        # sigma
        self.fc4 = nn.Linear(args.mlp_hidden_dim, args.action_dim_n[agent_id])
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            for lay in self.fc_hidden:
                orthogonal_init(lay)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, episode_limit, N, actor_input_dim), prob.shape(mini_batch_size, episode_limit, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        for lay in self.fc_hidden:
            x = self.activate_func(lay(x))
        mu = torch.tanh(self.fc3(x))
        log_sigma = -torch.relu(self.fc4(x))
        sigma = torch.exp(log_sigma)
        return mu, sigma


class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc_hidden = []
        for i in range(args.hidden_layers):
            self.fc_hidden.append(nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim))
        # self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            for lay in self.fc_hidden:
                orthogonal_init(lay)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, episode_limit, N, critic_input_dim), value.shape=(mini_batch_size, episode_limit, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        for lay in self.fc_hidden:
            x = self.activate_func(lay(x))
        value = self.fc3(x)
        return value


class MAPPO:
    def __init__(self, args, id):
        self.agent_id = id
        self.episode_limit = args.episode_limit
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_value_clip = args.use_value_clip

        # get the input dimension of actor and critic
        self.actor_input_dim = args.obs_dim_n[id]
        self.critic_input_dim = args.obs_dim_n
        # print(self.critic_input_dim)
        if self.use_rnn:
            print("------use rnn------")
            self.actor = Actor_RNN(args, self.agent_id, self.actor_input_dim)
            self.critic = Critic_RNN(args, sum(self.critic_input_dim))
        else:
            self.actor = Actor_MLP(args, self.agent_id, self.actor_input_dim)
            self.critic = Critic_MLP(args, sum(self.critic_input_dim))

        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.set_adam_eps:
            print("------set adam eps------")
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)

    def choose_action(self, obs_n, evaluate):
        with torch.no_grad():
            # print(obs_n)
            obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim)

            mu, sigma = self.actor(obs_n)  # prob.shape=(N,action_dim)
            dist = Normal(mu, sigma)
            a_n = dist.sample()
            a_logprob_n = dist.log_prob(a_n)
            return a_n.numpy(), a_logprob_n.numpy()

    def get_value(self, s):
        with torch.no_grad():
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            # s=np.hstack(s)
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)  # (state_dim,)-->(N,state_dim)
            v_n = self.critic(s)  # v_n.shape(N,1)
            return v_n.numpy().flatten()

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()  # get training data

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and td_target have no gradient
            # print(batch['r_n'][:, :, self.agent_id])
            # print(batch['v_n'][:, 1:][:, :, self.agent_id])
            deltas = batch['r_n'][:, :, self.agent_id] + self.gamma * batch['v_n'][:, 1:][:, :, self.agent_id] * (
                    1 - batch['done_n'][:, :, self.agent_id]) - batch['v_n'][:, :-1][:, :, self.agent_id]
            # deltas.shape=(batch_size,episode_limit,N)
            # print(deltas)
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,episode_limit,N)
            # print(adv)
            v_target = adv + batch['v_n'][:, :-1][:, :, self.agent_id]  # v_target.shape(batch_size,episode_limit,N)
            if self.use_adv_norm:  # Trick 1: advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        actor_inputs, critic_inputs = self.get_inputs(batch)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                """
                    get probs_now and values_now
                    probs_now.shape=(mini_batch_size, episode_limit, N, action_dim)
                    values_now.shape=(mini_batch_size, episode_limit, N)
                """
                # if self.use_rnn:
                #     # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                #     self.actor.rnn_hidden = None
                #     self.critic.rnn_hidden = None
                #     probs_now, values_now = [], []
                #     for t in range(self.episode_limit):
                #         prob = self.actor(actor_inputs[index, t].reshape(self.mini_batch_size * self.N,
                #                                                          -1))  # prob.shape=(mini_batch_size*N, action_dim)
                #         probs_now.append(
                #             prob.reshape(self.mini_batch_size, self.N, -1))  # prob.shape=(mini_batch_size,N,action_dim）
                #         v = self.critic(critic_inputs[index, t].reshape(self.mini_batch_size * self.N,
                #                                                         -1))  # v.shape=(mini_batch_size*N,1)
                #         values_now.append(v.reshape(self.mini_batch_size, self.N))  # v.shape=(mini_batch_size,N)
                #     # Stack them according to the time (dim=1)
                #     probs_now = torch.stack(probs_now, dim=1)
                #     values_now = torch.stack(values_now, dim=1)
                # else:
                #     probs_now = self.actor(actor_inputs[index])
                #     values_now = self.critic(critic_inputs[index]).squeeze(-1)
                mu, sigma = self.actor(actor_inputs[index])
                # print(mu,sigma)
                values_now = self.critic(critic_inputs[index]).squeeze(-1)
                # print(values_now)

                dist_now = Normal(mu, sigma)
                # print(dist_now.entropy())
                dist_entropy = dist_now.entropy().sum(2)
                # print(dist_entropy)
                # print(dist_entropy)
                a_logprob_n_now = dist_now.log_prob(batch['a_%d' % self.agent_id][index]).sum(2)
                # print(a_logprob_n_now)
                # print(a_logprob_n_now.sum(2))
                # print(batch['a_logprob_n'])
                # print(batch['a_logprob_n'][index, :, self.agent_id])
                # print(a_logprob_n_now - batch['a_logprob_n'][index, :, self.agent_id].detach())
                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index, :, self.agent_id].detach())
                # print(ratios)
                # print(adv[index])
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                # print(dist_entropy)
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1][:, :, self.agent_id].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - \
                                        v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    # print(values_now.squeeze(-1))
                    # print(v_target.shape)
                    # print(values_now.squeeze(-1).shape)
                    critic_loss = (values_now.squeeze(-1) - v_target[index]) ** 2

                self.ac_optimizer.zero_grad()
                # print(critic_loss)
                # print(critic_loss.mean())
                ac_loss = actor_loss.mean() + critic_loss.mean()
                ac_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_%d' % self.agent_id])
        critic_inputs.append(batch['s'])
        # print(critic_inputs)

        # if self.add_agent_id:
        #     # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
        #     agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.episode_limit,
        #                                                                           1, 1)
        #     actor_inputs.append(agent_id_one_hot)
        #     critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat([x for x in actor_inputs],
                                 dim=-1)  # actor_inputs.shape=(batch_size, episode_limit, N, actor_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs],
                                  dim=-1)  # critic_inputs.shape=(batch_size, episode_limit, N, critic_input_dim)
        # print(critic_inputs)
        return actor_inputs, critic_inputs

    def save_model(self, log_path_a, log_path_b):
        torch.save(self.actor.state_dict(), log_path_a)
        torch.save(self.critic.state_dict(), log_path_b)

    def load_model(self, log_path_a, log_path_b):
        self.actor.load_state_dict(torch.load(log_path_a))
        self.critic.load_state_dict(torch.load(log_path_b))
