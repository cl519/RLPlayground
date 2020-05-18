import random
import gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
class ContinuousCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width /world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()

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
            nn.Tanh(),
            nn.Linear(400, 300),
            nn.Tanh(),
            nn.Linear(300, num_actions)
        )

    def forward(self, state):
        # print('self.fc(state): ', self.fc(state))
        # print('F.tanh(self.fc(state)): ', torch.tanh(self.fc(state)))
        # print('torch.tensor(self.action_space.high[0]) * (F.tanh(self.fc(state)))', torch.tensor(self.action_space.high[0]) * (torch.tanh(self.fc(state))))
        return torch.tensor(self.action_space.high[0]) * (torch.tanh(self.fc(state)))

class CriticNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_inputs + num_actions, 400),
            nn.Tanh(),
            nn.Linear(400, 300),
            nn.Tanh(),
            nn.Linear(300, 1)
        )

    def forward(self, state, action):
        return self.fc(torch.cat([state, action], 1))

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

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

    def act(self, obs):
        with torch.no_grad():
            return self.actor(obs).numpy()

    def update(self, rollouts, train_iter):
        data = rollouts.batch_sampler(self.batch_size)
        s, d, a, r, o, o2 = data

        QValue = self.critic(o, a)
        with torch.no_grad():
            na = self.actor_target(o)
            Next_QValue = self.critic_target(o, na)
            QValPrime = r + self.gamma * (1-d) * Next_QValue.view(-1)

        # print('QValue shape: ', QValue.shape)
        # print('QValPrime shape: ', QValPrime.shape)
        critic_loss = ((QValue.view(-1) - QValPrime)**2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # writer.add_scalar('CriticLoss/train', critic_loss, train_iter)

        for p in self.critic.parameters():
            p.requires_grad = False

        policy_loss = -self.critic(o, self.actor(o)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        # writer.add_scalar('PolicyLoss/train', policy_loss, train_iter)


        with torch.no_grad():
            for target_param, self_param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1-self.polyak) * self_param.data)

            for target_param, self_param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1-self.polyak) * self_param.data)

def train():
    # env = gym.make('MountainCarContinuous-v0')
    env = ContinuousCartPoleEnv();

    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    action_space = env.action_space
    rollouts = ReplayBuffer()

    np.random.seed(31)
    torch.manual_seed(31)
    env.seed(31)
    random.seed(31)

    policy = DDPGAgent(obs_size, num_actions, env.action_space)
    # noise = OUNoise(env.action_space)

    o, ep_ret, run, ep_len = env.reset(), 0, 0, 0

    for j in range(60000):
        print("j: ", j)

        a = policy.act(torch.tensor(o, dtype=torch.float32))
        a += 0.1 * np.random.randn(num_actions)
        a = np.clip(a, action_space.low[0], action_space.high[0])

        env.render()
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        d = False if ep_len == 1000 else d

        rollouts.insert((j, d, a, r, o, o2))

        o = o2
        if d or (ep_len == 1000):
            run += 1
            # writer.add_scalar('TotalRewardPerEpisode/train', ep_ret, run)
            o, ep_ret, ep_len = env.reset(), 0, 0

        if len(rollouts.buffer) >= 64:
            policy.update(rollouts, j)


if __name__ == "__main__":
    train()