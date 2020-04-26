import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import time
import collections
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class RolloutStorage():
    def __init__(self, obs_size, BUFFER_LIMIT = 8000): #used to be 1000000
        self.buffer = collections.deque(maxlen = BUFFER_LIMIT)
        self.obs_size = obs_size

    def insert(self, transition):
        self.buffer.append(transition)
    def batch_sampler(self, batch_size = 200, get_old_log_probs=False):

        mini_batch = random.sample(self.buffer, batch_size)
        step_list, done_list, action_list, reward_list, prev_obs_list, obs_list = [], [], [], [], [], []

        for transition in mini_batch:
            s, d, a, r, po, o = transition
            step_list.append(s)
            done_list.append(d)
            action_list.append(a)
            #print('a: ', a)
            reward_list.append(r)
            prev_obs_list.append(po)
            obs_list.append(o)
            #print('po: ', po)
            #print('o: ', o)
        return torch.tensor(step_list, dtype=torch.float), torch.tensor(done_list), \
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

    def sample(self, state):# TODO understand why you need this
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * torch.tensor(self.action_scale) + self.action_bias #need detach?
        log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(torch.tensor(self.action_scale) * (1-y_t.pow(20) + epsilon)) #added casting to action_scale to tensor
        #print("log_prob: ", log_prob)
        #log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * torch.tensor(self.action_scale) + self.action_bias #need detach here too?
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
        return self.fc(torch.cat([state, action], 1))

class SAC:
    def __init__(self, num_inputs, num_actions, action_space, hidden_dim = 128, batch_size = 64, policy_epochs = 8,
                 entropy_coef=0.001, gamma=0.99, tau = 0.001, alpha = 0.2):
        #super().__init__(num_inputs, num_actions, hidden_dim, learning_rate, batch_size, policy_epochs, entropy_coef)
        self.batch_size = batch_size

        self.actor = ActorNetwork(num_inputs, num_actions, hidden_dim, action_space)
        self.actor_target = ActorNetwork(num_inputs, num_actions, hidden_dim, action_space)

        self.Q1 = CriticNetwork(num_inputs, num_actions, hidden_dim)
        self.Q2 = CriticNetwork(num_inputs, num_actions, hidden_dim)
        self.Q1_target = CriticNetwork(num_inputs, num_actions, hidden_dim)
        self.Q2_target = CriticNetwork(num_inputs, num_actions, hidden_dim)
        self.critic_criterion = nn.MSELoss()

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=1e-3)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=1e-3)

    def act(self, obs):
        # state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        # action = self.actor.forward(state)
        # action = action.squeeze(0).cpu().detach().numpy()


        #obs = Variable(torch.from_numpy(obs).float().unsqueeze(0))
        action, _, _ = self.actor.sample(obs)
        #print("action actor returns: ", action)
        #action = action.detach().numpy()[0,0]
        action = action.detach().numpy() #TODO??? remove [0]
        #print("action after detach numpy and[0]: ", action)
        return action

    def update(self, rollouts, train_iter):
        #for epoch in range(self.policy_epochs):
        data = rollouts.batch_sampler(self.batch_size, get_old_log_probs=False)


        _, done_batch, actions_batch, returns_batch, prev_obs_batch, obs_batch = data

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(obs_batch) #WE USE ACTOR, NOT ACTOR TARGET!
            Q1_next_target = self.Q1_target(obs_batch, next_state_action)
            Q2_next_target = self.Q2_target(obs_batch, next_state_action)
            min_next_target = torch.min(Q1_next_target, Q2_next_target) - self.alpha * next_state_log_pi #How do I get this log pi? What information does this log pi give us?
            next_Qvalue = returns_batch + (1-done_batch) * self.gamma * min_next_target
        pi, log_pi, _ = self.actor.sample(prev_obs_batch)

        Q1_pi = self.Q1(prev_obs_batch, pi)
        Q2_pi = self.Q2(prev_obs_batch, pi)

        Q1_loss = self.critic_criterion(Q1_pi, next_Qvalue)
        Q2_loss = self.critic_criterion(Q2_pi, next_Qvalue)



        policy_loss = -(torch.min(Q1_pi, Q2_pi) - self.alpha * log_pi).mean()


        #print("Q1_loss: ", Q1_loss)
        self.Q1_optimizer.zero_grad()
        Q1_loss.backward(retain_graph=True)#need retain graph?
        self.Q1_optimizer.step()
        writer.add_scalar('Q1Loss/train', Q1_loss, train_iter)

        self.Q2_optimizer.zero_grad()
        Q2_loss.backward(retain_graph=True)
        self.Q2_optimizer.step()

        writer.add_scalar('Q2Loss/train', Q2_loss, train_iter)

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        writer.add_scalar('PolicyLoss/train', policy_loss, train_iter)

        for target_param, self_param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            target_param.data.copy_(self_param.data * self.tau + target_param.data * (1 - self.tau))

        for target_param, self_param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            target_param.data.copy_(self_param.data * self.tau + target_param.data * (1 - self.tau))

def train():

    #env = gym.make('HalfCheetah-v2')
    env = gym.make('MountainCarContinuous-v0')
    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    rollouts = RolloutStorage(obs_size)

    #SETTING SEED: it is good practice to set seeds when running experiments to keep results comparable
    np.random.seed(123)
    torch.manual_seed(123)
    env.seed(123)
    random.seed(123)

    policy = SAC(obs_size, num_actions, env.action_space) #can add  env.action_space if you want to scale by max environment action value



    done = False;
    prev_obs = env.reset()
    prev_obs = torch.tensor(prev_obs, dtype=torch.float32)
    eps_reward = 0.
    run = 0
    episode_steps = 0
    for j in range(60000):

        print("j: ", j)
        if done:

            writer.add_scalar('TotalRewardPerEpisode/train', eps_reward, run)

            run+=1


            # Reset Environment
            done = False
            prev_obs = env.reset()
            prev_obs = torch.tensor(prev_obs, dtype=torch.float32)
            obs = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32)
            eps_reward = 0.
            episode_steps = 0
        else:
            obs = prev_obs
        env.render()
        action = policy.act(obs)
        #print("action: ", action)
        obs, reward, done, info = env.step(action) #action.item()
        #print('obs type: ', type(obs))

        episode_steps += 1

        done = torch.tensor(done, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        reward = reward
        rollouts.insert((j, done, action, reward, prev_obs.numpy(), obs)) #obs is numpy array

        prev_obs = torch.tensor(obs, dtype=torch.float32)
        eps_reward += reward
        if len(rollouts.buffer) >= 64:
            policy.update(rollouts, j)


if __name__ == "__main__":
    train()
