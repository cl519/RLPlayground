import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import argparse
import os

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()



def get_options():
    parser = argparse.ArgumentParser(description='Clone some expert data..')
    parser.add_argument('bc_data', type=str,
        help="The main datastore for this particular expert.")

    args = parser.parse_args()
    return args

def process_data(bc_data_dir):
    """
    Runs training for the agent.
    """
    # Load the file store.
    # In the future (TODO) move this to a seperate thread.
    print('bc_data_dir: ', bc_data_dir)
    states, actions = [], []
    shards = [x for x in os.listdir(bc_data_dir) if x.endswith('.npy')]
    print("Processing shards: {}".format(shards))
    for shard in shards:
        shard_path = os.path.join(bc_data_dir, shard)
        with open(shard_path, 'rb') as f:
            data = np.load(f, allow_pickle = True, encoding='latin1')
            shard_states, unprocessed_actions = zip(*data)
            shard_states = [x.flatten() for x in shard_states]

            # Add the shard to the dataset
            states.extend(shard_states)
            actions.extend(unprocessed_actions)


    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32) #/2
    print("Processed with {} pairs".format(len(states)))
    return states, actions

class Net(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.fc = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, state):
        logits = self.fc(state)
        # print('logits shape: ', logits.shape)
        return logits, torch.argmax(logits, dim=1)

class Policy():
    def __init__(self, num_inputs, num_actions):
        self.net = Net(num_inputs, num_actions)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def update(self, expert_state_batch, expert_action_batch, iter_num):

        self.optimizer.zero_grad()
        # "outputs" is logits
        outputs, _ = self.net(torch.tensor(expert_state_batch, dtype=torch.float32))

        loss = self.criterion(outputs, torch.tensor(expert_action_batch, dtype=torch.long))
        print("loss: ", loss)
        writer.add_scalar('Loss/train', loss, iter_num)
        loss.backward()
        self.optimizer.step()

    def act(self, obs):
        with torch.no_grad():
            # print('self.net(obs): ', self.net(obs))
            _, action = self.net(obs)
        return action.numpy()

def train(opts):
    expert_state, expert_action = process_data(opts.bc_data)


    env = gym.make('MountainCar-v0')
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n

    policy = Policy(num_inputs, num_actions)


    num_batch = 4094//64
    num_epoch = 50
    iter_num = 0

    for _ in range(num_epoch):
        for _ in range(num_batch):
            batch_index = np.random.choice(len(expert_state), 64)
            expert_state_batch, expert_action_batch = expert_state[batch_index], expert_action[batch_index]


            policy.update(expert_state_batch, expert_action_batch, iter_num)
            iter_num += 1

    # print('EVALUATION')

    # o = env.reset()
    # d = False
    # for _ in range(10):
    #     while(not d):
    #         print('o: ', o)
    #         a = policy.act(torch.tensor([o], dtype=torch.float32))
    #         print('a: ', a)
    #         env.render()
    #         o2, r, d, _ = env.step(a.item())
    #         o = o2
    #     o, d = env.reset(), False

    # env.close()



if __name__ == "__main__":
    opts = get_options()
    train(opts)