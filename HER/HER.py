import argparse
import random
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
import wandb
wandb.init(project="sac")

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class RolloutStorage():
    def __init__(self, obs_size, BUFFER_LIMIT = int(1e6)): #used to be 1000000 #8000
        self.buffer = collections.deque(maxlen = BUFFER_LIMIT)
        self.obs_size = obs_size

    def insert(self, transition):
        self.buffer.append(transition)
    def batch_sampler(self, batch_size = 200):

        mini_batch = random.sample(self.buffer, batch_size)
        done_list, action_list, reward_list, prev_obs_list, obs_list = [], [], [], [], [] #step_list

        # print(mini_batch)

        for transition in mini_batch:
            d, a, r, po, o = transition
            done_list.append(d)
            action_list.append(a)
            reward_list.append(r)
            prev_obs_list.append(po)
            obs_list.append(o)
        return torch.tensor(done_list, dtype=torch.float), \
                   torch.tensor(action_list, dtype=torch.float), torch.tensor(reward_list, dtype=torch.float), \
                   torch.tensor(prev_obs_list, dtype=torch.float), torch.tensor(obs_list, dtype=torch.float)




class ActorNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space):
        super().__init__()
        # print("num_inputs type: ", type(num_inputs))
        # print("num_actions type: ", type(num_actions))
        # print("hidden_dim type: ", type(hidden_dim))
        self.action_scale = action_space.high
        self.action_bias = 0
        self.fc = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):
        x = self.fc(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max = LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * torch.tensor(self.action_scale) + self.action_bias

        log_prob = normal.log_prob(x_t).sum(axis=-1)



        log_prob -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t))).sum(1)

        mean = torch.tanh(mean) * torch.tensor(self.action_scale) + self.action_bias
        return action, log_prob, mean

class CriticNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return torch.squeeze(self.fc(torch.cat([state, action], 1)), -1)

class SAC: #fixed entropy regularization
    def __init__(self, num_inputs, num_actions, action_space, hidden_dim = 128, batch_size = 100, policy_epochs = 8,
                 entropy_coef=0.001, gamma=0.99, tau = 0.005, alpha = 0.2):
        self.batch_size = batch_size

        self.actor = ActorNetwork(num_inputs, num_actions, hidden_dim, action_space)

        self.Q1 = CriticNetwork(num_inputs, num_actions, hidden_dim)
        self.Q2 = CriticNetwork(num_inputs, num_actions, hidden_dim)
        self.Q1_target = CriticNetwork(num_inputs, num_actions, hidden_dim)
        self.Q2_target = CriticNetwork(num_inputs, num_actions, hidden_dim)
        self.critic_criterion = nn.MSELoss()

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        for target_param, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=1e-4)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=1e-4)

    def act(self, obs, evaluation=False):
        if evaluation is False:
            action, _, _ = self.actor.sample(obs)
        else:
            _, _, action = self.actor.sample(obs)
        return action.detach().numpy()[0]

    def update(self, rollouts, train_iter):
        data = rollouts.batch_sampler(self.batch_size)


        done_batch, actions_batch, returns_batch, prev_obs_batch, obs_batch = data

        # self.Q1_optimizer.zero_grad()
        # self.Q2_optimizer.zero_grad()


        # Q1 = self.Q1(prev_obs_batch, actions_batch)
        # Q2 = self.Q2(prev_obs_batch, actions_batch)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(obs_batch) #actor, not actor taret
            Q1_next_target = self.Q1_target(obs_batch, next_state_action)
            Q2_next_target = self.Q2_target(obs_batch, next_state_action)
            # print('Q1_next_target.shape: ', Q1_next_target.shape)
            # print('next_state_log_pi.shape: ', next_state_log_pi.shape)
            min_next_target = torch.min(Q1_next_target, Q2_next_target) - self.alpha * next_state_log_pi
            # print('min_next_target.shape: ', min_next_target.shape)
            # print('returns_batch.shape: ', returns_batch.shape)
            # print('done_batch.shape: ', done_batch.shape)
            next_Qvalue = returns_batch + (1-done_batch) * self.gamma * min_next_target
            # print('next_QValue.shape: ', next_Qvalue.shape)


        Q1 = self.Q1(prev_obs_batch, actions_batch)
        Q2 = self.Q2(prev_obs_batch, actions_batch)


        Q1_loss = nn.MSELoss()(Q1, next_Qvalue.detach())

        # Q1_loss = 0.5 * (Q1 - next_Qvalue).pow(2).mean()

        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.Q1_optimizer.step()

        Q2_loss = self.critic_criterion(Q2, next_Qvalue)

        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.Q2_optimizer.step()

        pi, log_pi, _ = self.actor.sample(prev_obs_batch)

        Q1_pi = self.Q1(prev_obs_batch, pi)
        Q2_pi = self.Q2(prev_obs_batch, pi)

        # temp = torch.min(Q1_pi, Q2_pi)
        policy_loss = -(torch.min(Q1_pi, Q2_pi) - self.alpha * log_pi).mean()

        # for p in q_params:
        #     p.requires_grad = False

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # for p in q_params:
        #     p.requires_grad = True

        wandb.log({'Q1Loss': Q1_loss, 'Q2Loss':Q2_loss, 'PolicyLoss':policy_loss, 'train_iter': train_iter})


        # writer.add_scalar('PolicyLoss/train', policy_loss, train_iter)

        for target_param, self_param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            target_param.data.copy_(self_param.data * self.tau + target_param.data * (1 - self.tau))

        for target_param, self_param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            target_param.data.copy_(self_param.data * self.tau + target_param.data * (1 - self.tau))

def train():

    env = gym.make('FetchReach-v1') #max_ep_len is 50
    print('env observation space: ', env.observation_space)
    print('env action space: ', env.action_space)
    obs_size = env.observation_space['observation'].shape[0]
    goal_size = env.observation_space['desired_goal'].shape[0]
    num_actions = env.action_space.shape[0]
    rollouts = RolloutStorage(obs_size)

    #SETTING SEED: it is good practice to set seeds when running experiments to keep results comparable
    np.random.seed(234)
    torch.manual_seed(234)
    env.seed(234)
    random.seed(234)

    # print('env.action_space sample: ', env.action_space.sample())
    # print('env.observation_space sample: ', env.observation_space.sample())

    policy = SAC(obs_size + goal_size, num_actions, env.action_space)

    o, ep_ret, run, ep_len = env.reset(), 0, 0, 0
    d = False
    for i in range(400): #400 episodes

        print(i, ' episodes')

        desired_goal = o['desired_goal']
        traj = []
        while not d:
            a = policy.act(torch.tensor(np.concatenate((o['observation'], desired_goal)), dtype=torch.float32).unsqueeze(0))
            # a = env.action_space.sample() #append with desired goal

            env.render()
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            d = False if ep_len == 50 else d #1000

            ag = o2['achieved_goal']
            traj.append((d, a, r, o, o2, ag))

            rollouts.insert((d, a, r, np.concatenate((o['observation'], desired_goal)), np.concatenate((o2['observation'], desired_goal))))

            if d or (ep_len == 50):
                wandb.log({'TotalRewardPerEpisode': ep_ret, 'episode_step': run})
                run += 1
                break

            o = o2

        substitute_goal = o['achieved_goal']
        for transition in traj:
            d, a, r, o, o2, ag = transition
            substitute_reward = env.compute_reward(ag, substitute_goal, _)
            rollouts.insert((d, a, substitute_reward, np.concatenate((o['observation'], substitute_goal)), np.concatenate((o2['observation'], substitute_goal))))

        if len(rollouts.buffer) >= 100:
            for t in range(50):
                policy.update(rollouts, i)

        o, ep_ret, ep_len, d = env.reset(), 0, 0, False
        # if d or (ep_len == 1000):
        #     # wandb.log({'TotalRewardPerEpisode': ep_ret, 'episode_step': run})
        #     run += 1
        #     o, ep_ret, ep_len = env.reset(), 0, 0
        #

    env.close()

    # if not os.path.isdir('model'):
    #     os.makedirs('model')
    #
    # torch.save(policy.actor.state_dict(), 'model/actor_param')

def test():
    env = gym.make('HalfCheetah-v2')
    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    hidden_dim = 128


    policy = SAC(obs_size, num_actions, env.action_space)

    policy.actor = ActorNetwork(obs_size, num_actions, hidden_dim, env.action_space)

    policy.actor.load_state_dict(torch.load('model/actor_param'))

    np.random.seed(345)
    torch.manual_seed(345)
    env.seed(345)
    random.seed(345)

    o, ep_ret, run, ep_len = env.reset(), 0, 0, 0

    for i in range(10):
        print(i, ' runs')
        d = False
        while not d or (ep_len == 1000):
            a = policy.act(torch.tensor(o, dtype=torch.float32).unsqueeze(0), True)
            env.render()
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            o = o2
            if d:
                # wandb.log({'TotalRewardPerEpisode/test': ep_ret})
                run += 1
                o, ep_ret, ep_len = env.reset(), 0, 0
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    mode = args.mode

    if mode == 'train':
        train()
    elif mode == 'test':
        test()
