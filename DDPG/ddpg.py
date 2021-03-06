import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
from torch.utils.tensorboard import SummaryWriter
import argparse
import os

writer = SummaryWriter()

import copy
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size=1, mu=0, theta=0.05, sigma=0.25):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    def __init__(self, BUFFER_LIMIT = 8000):
        self.buffer = collections.deque(maxlen = BUFFER_LIMIT)

    def insert(self, transition):
        self.buffer.append(transition)

    def batch_sampler(self, batch_size = 64):
        mini_batch = random.sample(self.buffer, batch_size)
        step_list, done_list, action_list, reward_list, o_list, o2_list = [], [], [], [], [], []
        for transition  in mini_batch:
            s, d, a, r, o, o2 = transition
            step_list.append(s)
            done_list.append(d)
            action_list.append(a)
            reward_list.append(r)
            o_list.append(o)
            o2_list.append(o2)
        return torch.tensor(step_list, dtype=torch.float), torch.tensor(done_list, dtype=torch.float), torch.tensor(action_list, dtype=torch.float), \
                    torch.tensor(reward_list, dtype=torch.float), torch.tensor(o_list, dtype=torch.float), torch.tensor(o2_list, dtype=torch.float)

class ActorNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, action_space):
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
        return torch.tensor(self.action_space.high[0]) * (torch.tanh(self.fc(state)))

class CriticNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_inputs + num_actions, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, state, action):
        q = self.fc(torch.cat([state, action], 1))
        return torch.squeeze(q, -1)

class DDPGAgent:
    def __init__(self, num_inputs, num_actions, action_space):
        self.batch_size = 64

        self.actor = ActorNetwork(num_inputs, num_actions, action_space)
        self.actor_target = ActorNetwork(num_inputs, num_actions, action_space)

        self.critic = CriticNetwork(num_inputs, num_actions)
        self.critic_target = CriticNetwork(num_inputs, num_actions)

        self.gamma = 0.99
        self.polyak = 0.995

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3) 
        self.critic_criterion = nn.MSELoss()

    def act(self, obs):
        with torch.no_grad():
            return self.actor(obs).numpy()

    def update(self, rollouts, train_iter):
        data = rollouts.batch_sampler(self.batch_size)
        s, d, a, r, o, o2 = data

        QValue = self.critic(o, a)
        with torch.no_grad():
            na = self.actor_target(o2)
            Next_QValue = self.critic_target(o2, na)
            QValPrime = r + self.gamma * (1-d) * Next_QValue
        critic_loss = self.critic_criterion(QValue, QValPrime.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        writer.add_scalar('CriticLoss/train', critic_loss, train_iter)

        for p in self.critic.parameters():
            p.requires_grad = False

        policy_loss = -self.critic(o, self.actor(o)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        writer.add_scalar('PolicyLoss/train', policy_loss, train_iter)

        with torch.no_grad():
            for target_param, self_param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1-self.polyak) * self_param.data)

            for target_param, self_param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1-self.polyak) * self_param.data)

def train():
    env = gym.make('MountainCarContinuous-v0') 

    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    action_space = env.action_space
    rollouts = ReplayBuffer()

    np.random.seed(2)
    torch.manual_seed(2)
    env.seed(2)
    random.seed(2)

    policy = DDPGAgent(obs_size, num_actions, env.action_space)
    noise = OUNoise(num_actions)

    o, ep_ret, run, ep_len = env.reset(), 0, 0, 0

    for j in range(100000):
        print(j, ' transitions')

        a = policy.act(torch.tensor(o, dtype=torch.float32))
        noi = noise.sample()
        a = np.clip(a + noi, -1, 1) #a*.2


        # Gaussian Noise
        # a += 0.1 * np.random.randn(num_actions)
        # a = np.clip(a, action_space.low[0], action_space.high[0])

        env.render()
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        d = False if ep_len == 1000 else d

        rollouts.insert((j, d, a, r, o, o2))

        o = o2
        if d or (ep_len == 1000):
            writer.add_scalar('TotalRewardPerEpisode/train', ep_ret, run)
            run += 1
            o, ep_ret, ep_len = env.reset(), 0, 0

        if len(rollouts.buffer) >= 64:
            policy.update(rollouts, j)

    env.close()


    if not os.path.isdir('model'):
        os.makedirs('model')

    torch.save(policy.actor.state_dict(), 'model/actor_param')
    torch.save(policy.actor_target.state_dict(), 'model/actor_target_param')
    torch.save(policy.critic.state_dict(), 'model/critic_param')
    torch.save(policy.critic_target.state_dict(), 'model/critic_target_param')

def test():
    env = gym.make('MountainCarContinuous-v0')
    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    policy = DDPGAgent(obs_size, num_actions, env.action_space)
    #model.load.state_dict(torch.load(PATH))
    hidden_dim = 128

    policy.actor = ActorNetwork(obs_size, num_actions, env.action_space)
    policy.actor_target = ActorNetwork(obs_size, num_actions, env.action_space)
    policy.critic = CriticNetwork(obs_size, num_actions)
    policy.critic_target = CriticNetwork(obs_size, num_actions)

    policy.actor.load_state_dict(torch.load('model/actor_param'))
    policy.actor_target.load_state_dict(torch.load('model/actor_target_param'))
    policy.critic.load_state_dict(torch.load('model/critic_param'))
    policy.critic_target.load_state_dict(torch.load('model/critic_target_param'))

    np.random.seed(2)
    torch.manual_seed(2)
    env.seed(2)
    random.seed(2)

    o, ep_ret, run = env.reset(), 0, 0

    for i in range(100):
        d = False
        while not d:
            a = policy.act(torch.tensor(o, dtype=torch.float32))
            env.render()
            o2, r, d, _ = env.step(a)
            ep_ret += r

            o = o2
            if d:
                writer.add_scalar('TotalRewardPerEpisode/test', ep_ret, run)
                run += 1
                o, ep_ret = env.reset(), 0
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
