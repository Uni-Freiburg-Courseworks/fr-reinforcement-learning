import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from gridworld import GridworldEnv
import numpy as np
from tqdm import tqdm

optimal_policy = [0, 3, 3, 2, 0, 0, 0, 2, 0, 0, 1, 2, 0, 1, 1, 0]


def get_action(s):
    return optimal_policy[s]


def tt(ndarray):
    # Untoggle to use CUDA
    # return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
    return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)


class Network(nn.Module):
    def __init__(self, state_dim, non_linearity=F.relu, hidden_dim=10):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self._non_linearity = non_linearity

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)

    def init_zero(self):
        torch.nn.init.constant_(self.fc1.weight, 0)
        torch.nn.init.constant_(self.fc1.bias, 0)
        torch.nn.init.constant_(self.fc2.weight, 0)
        torch.nn.init.constant_(self.fc2.bias, 0)
        torch.nn.init.constant_(self.fc3.weight, 0)
        torch.nn.init.constant_(self.fc3.bias, 0)


def overwrite_params(net_z, net_v, gamma, trace_decay):
    for param_z, param_v in zip(net_z.parameters(), net_v.parameters()):
        param_z.data.copy_(gamma * trace_decay * param_z.data + param_v.grad.data)

def update_params(net_v, net_z, alpha, delta):
    for param_v, param_z in zip(net_v.parameters(), net_z.parameters()):
        param_v.data.copy_(param_v.data + alpha * delta * param_z.data)

class TDLambda:
    def __init__(self, state_dim, gamma, trace_decay, alpha):
        self.v = Network(state_dim)
        self.z = Network(state_dim)

        # Untoggle to use CUDA
        # self.v.cuda()
        # self.z.cuda()
        self.gamma = gamma
        self.trace_decay = trace_decay
        self.alpha = alpha

        self.z.init_zero()

    def train(self, episodes, time_steps):
        for e in tqdm(range(episodes)):
            s = env.reset()

            # TODO: Implement this
            # Initialize z each episode
            self.z.init_zero()

            for t in range(time_steps):
                a = get_action(s)
                ns, r, d = env.step(a)

                # TODO: Implement this
                self.v.zero_grad()
                self.v(tt(np.array([s]))).backward() # why do we need .mean() here? for multiple batch?

                overwrite_params(self.z, self.v, self.gamma, self.trace_decay)
                delta = tt(np.array([r])) + (1-d) * self.gamma * self.v(tt(np.array([ns]))) - self.v(tt(np.array([s])))
                update_params(self.v, self.z, self.alpha, delta)

                if d:
                    break

                s = ns

        return self.v


if __name__ == "__main__":
    env = GridworldEnv()
    env.seed(0)
    tdlambda = TDLambda(1, gamma=0.99, trace_decay=0.5, alpha=0.001)

    episodes = 100000
    time_steps = 50

    w = tdlambda.train(episodes, time_steps)

    for i in range(env.nS):
        if i % 4 == 0:
            print('\n')
        print("%s " % (w(tt(np.array([i]))).item()), end='')
