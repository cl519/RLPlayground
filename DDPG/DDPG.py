import random
import gym
import numpy as np
from collections import deque
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
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

#reset noise at the beginning of episode
#num of episodes:

def GNoise(action, action_space):
    # print("action before noise: ", action)
    action += 0.1 * np.random.randn(action_space.shape[0])
    # print("action after noise: ", action)
    ret = np.clip(action, -action_space.high, action_space.high)
    # print("action after clipping: ", ret)
    return ret

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.2, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

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
            reward_list.append(r)
            prev_obs_list.append(po)
            obs_list.append(o)
        return torch.tensor(step_list, dtype=torch.float), torch.tensor(done_list), \
                   torch.tensor(action_list, dtype=torch.float), torch.tensor(reward_list, dtype=torch.float), \
                   torch.tensor(prev_obs_list, dtype=torch.float), torch.tensor(obs_list, dtype=torch.float)




class ActorNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, action_space, hidden_dim):
        super().__init__()
        self.num_actions = num_actions
        self.action_space = action_space
        self.fc = nn.Sequential(
            nn.Linear(num_inputs, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, num_actions)
        )

    def forward(self, state):
        return torch.tensor(self.action_space.high) * (F.tanh(self.fc(state)))

class CriticNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()
        # self.fc1 = nn.Sequential(
        #     nn.Linear(num_inputs, 400),
        #     nn.ReLU(),
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(400 + num_actions, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, num_actions)
        # )

        self.fc = nn.Sequential(
            nn.Linear(num_inputs + num_actions, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, num_actions)
        )

    def forward(self, state, action):
        # x = self.fc1(state)
        # return self.fc2(torch.cat([x, action], 1))
        return self.fc(torch.cat([state, action], 1))

class DDPGAgent:
    def __init__(self, num_inputs, num_actions, action_space, hidden_dim = 128, batch_size = 64, policy_epochs = 8,
                 entropy_coef=0.001, gamma=0.95, tau = 0.001):
        #super().__init__(num_inputs, num_actions, hidden_dim, learning_rate, batch_size, policy_epochs, entropy_coef)
        self.policy_epochs = policy_epochs
        self.batch_size = batch_size

        self.actor = ActorNetwork(num_inputs, num_actions, action_space, hidden_dim)
        self.actor_target = ActorNetwork(num_inputs, num_actions, action_space, hidden_dim)

        self.critic = CriticNetwork(num_inputs, num_actions, hidden_dim)
        self.critic_target = CriticNetwork(num_inputs, num_actions, hidden_dim)
        self.critic_criterion = nn.MSELoss()

        self.gamma = gamma
        self.tau = tau

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3) #1e-3 blows up at the end, 1e-4 spiky the whole time
        #weight_decay=0.01

    def act(self, obs):
        # state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        # action = self.actor.forward(state)
        # action = action.squeeze(0).cpu().detach().numpy()


        #obs = Variable(torch.from_numpy(obs).float().unsqueeze(0))
        action = self.actor.forward(obs)
        # print('---------act--------')
        # print('action shape before detach: ', action.shape)
        # print('action type: ', type(action))
        # print('---------act--------')
        #print("action before detach numpy: ", action)
        action = action.detach().numpy()[0] #TODO???
        #print("action after detach numpy and[0]: ", action)
        return action

    def update(self, rollouts, train_iter):
        #for epoch in range(self.policy_epochs):
        data = rollouts.batch_sampler(self.batch_size, get_old_log_probs=False)


        _, done_batch, actions_batch, returns_batch, prev_obs_batch, obs_batch = data

        #USE TARGET NETWORKS
        #Critic Loss
        QValue = self.critic(prev_obs_batch, actions_batch)
        next_actions_batch = self.actor_target(obs_batch)
        Next_QValue = self.critic_target(obs_batch, next_actions_batch.detach())
        #print('returns_batch before unsqeeze: ', returns_batch.shape)
        #print('returns_batch after unsqqeze: ', returns_batch.unsqueeze((1)).shape)
        QValPrime = returns_batch.unsqueeze(1) + (1-done_batch)*self.gamma * Next_QValue #Added unsqueeze in order to broadcast
        critic_loss = self.critic_criterion(QValue, QValPrime.detach())

        self.critic_optimizer.zero_grad()
        #print("critic loss: ", critic_loss)
        critic_loss.backward()
        self.critic_optimizer.step()

        #writer.add_scalar('CriticLoss/train', critic_loss, train_iter)

        #Policy Loss
        policy_loss = -self.critic(prev_obs_batch, self.actor(prev_obs_batch)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # writer.add_scalar('PolicyLoss/train', policy_loss, train_iter)

        for target_param, self_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self_param.data * self.tau + target_param.data * (1 - self.tau))

        for target_param, self_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self_param.data * self.tau + target_param.data * (1 - self.tau))



def train():

    #
    # env = gym.make('HalfCheetah-v2')
    # for i_episode in range(20):
    #     observation = env.reset()
    #     for t in range(100):
    #         env.render()
    #         action = env.action_space.sample()
    #         print('env.action_space.shape', env.action_space.shape)
    #         print('env.action_space.shape[0]', env.action_space.shape[0])
    #         print('env.action_space.low', env.action_space.low)
    #         print('env.action_space.high', env.action_space.high)
    #         observation, reward, done, info = env.step(action)
    #         if done:
    #             print("Episode finished after {} timesteps".format(t + 1))
    #             break
    # env.close()

    env = gym.make('MountainCarContinuous-v0')
    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    rollouts = RolloutStorage(obs_size)

    #SETTING SEED: it is good practice to set seeds when running experiments to keep results comparable
    np.random.seed(345)
    torch.manual_seed(345)
    env.seed(345)
    random.seed(345)


    policy = DDPGAgent(obs_size, num_actions, env.action_space)
    noise = OUNoise(env.action_space)

    done = False
    prev_obs = env.reset()
    prev_obs = torch.tensor(prev_obs, dtype=torch.float32)
    eps_reward = 0.
    run = 0
    episode_steps = 0
    for j in range(60000):

        print("j: ", j)
        if done:

            # writer.add_scalar('TotalRewardPerEpisode/train', eps_reward, run)
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
        # action = noise.get_action(action, j)
        print("original action as array: ", np.asarray(action))
        print("asarray type: ", type(np.asarray(action)))
        print("asarray shape: ", np.asarray(action).shape)
        action = GNoise(action, env.action_space)
        print("action after GNoise: ", action)
        print("action type after GNoise: ", type(action))
        print("action shape: ", action.shape)

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



    if not os.path.isdir('model'):
        os.makedirs('model')
    print('ACTOR')
    for param_tensor in policy.actor.state_dict():
        print(param_tensor, "\t", policy.actor.state_dict()[param_tensor].size())
    print('ACTOR_TARGET')
    for param_tensor in policy.actor_target.state_dict():
        print(param_tensor, "\t", policy.actor_target.state_dict()[param_tensor].size())
    print('CRITIC')
    for param_tensor in policy.critic.state_dict():
        print(param_tensor, "\t", policy.critic.state_dict()[param_tensor].size())
    print('CRITIC_TARGET')
    for param_tensor in policy.actor.state_dict():
        print(param_tensor, "\t", policy.critic_target.state_dict()[param_tensor].size())


    torch.save(policy.actor.state_dict(), 'model/actor_param')
    torch.save(policy.actor_target.state_dict(), 'model/actor_target_param')
    torch.save(policy.critic.state_dict(), 'model/critic_param')
    torch.save(policy.critic_target.state_dict(), 'model/critic_target_param')


def eval():
    env = gym.make('MountainCarContinuous-v0')
    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    model = DDPGAgent(obs_size, num_actions)
    #model.load.state_dict(torch.load(PATH))
    hidden_dim = 128

    model.actor = ActorNetwork(obs_size, num_actions, hidden_dim)
    model.actor_target = ActorNetwork(obs_size, num_actions, hidden_dim)
    model.critic = CriticNetwork(obs_size, num_actions, hidden_dim)
    model.critic_target = CriticNetwork(obs_size, num_actions, hidden_dim)

    # print("Model's state_dict:")
    # for param_tensor in policy.actor.state_dict():
    #     print(param_tensor, "\t", policy.actor.state_dict()[param_tensor].size())

    model.actor.load_state_dict(torch.load('model/actor_param'))
    model.actor_target.load_state_dict(torch.load('model/actor_target_param'))
    model.critic.load_state_dict(torch.load('model/critic_param'))
    model.critic_target.load_state_dict(torch.load('model/critic_target_param'))

    np.random.seed(123)
    torch.manual_seed(123)
    env.seed(123)
    random.seed(123)

    # for j in range(0): #num of training iterations 50
    done = False;
    prev_obs = env.reset()
    prev_obs = torch.tensor(prev_obs, dtype=torch.float32)
    # run = 0
    for step in range(25000): #25000
        if done:
            # avg_eps_reward.update(eps_reward)
            #writer.add_scalar('TotalRewardPerEpisode/train', eps_reward, j * 25000 + step)
            # run+=1
            # Reset Environment
            obs = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32)
            #eps_reward = 0.
        else:
            obs = prev_obs
        env.render()
        action = model.act(obs)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, info = env.step(action) #action.item()

        done = torch.tensor(done, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        reward = reward
        #rollouts.insert((step, done, action, reward, prev_obs.numpy(), obs)) #obs is numpy array

        #prev_obs = torch.tensor(obs, dtype=torch.float32)
        #eps_reward += reward




if __name__ == "__main__":
    train()

