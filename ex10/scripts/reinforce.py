import sys
import argparse
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from collections import namedtuple
import time
import matplotlib.pyplot as plt
import pandas as pd
from mountain_car import MountainCarEnv

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tt(x):
    if isinstance(x, float):
        x = np.array([x])
    return Variable(torch.from_numpy(x).float().to(device), requires_grad=False)


class StateValueFunction(nn.Module):

    def __init__(self, state_dim, hidden_dim=20):
        super(StateValueFunction, self).__init__()
        # Implement this!
        self._net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Implement this!
        return self._net(x)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=20):
        super(Policy, self).__init__()

        self._net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax()
        )

    def forward(self, x):
        # Implement this!
        return self._net(x)

class REINFORCE:
    def __init__(self, state_dim, action_dim, gamma, step_size, use_baseline=False, step_size_v=None):

        self._pi = Policy(state_dim, action_dim).to(device)
        self._pi_optimizer = optim.Adam(self._pi.parameters(), lr=0.0001)

        if use_baseline:
            self._V = StateValueFunction(state_dim).to(device)
            self._V_optimizer = optim.Adam(self._V.parameters(), lr=0.0001)
            self._step_size_v = step_size_v
        else:
            self._V = None
        self._use_baseline = use_baseline

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._action_dim = action_dim
        self._step_size = step_size

    def get_action(self, s):
        # Implement this!
        a = self._pi(tt(s)).argmax(dim=0).item()

        return a

    def train(self, episodes, time_steps):
        stats = EpisodeStats(episode_lengths=np.zeros(
            episodes), episode_rewards=np.zeros(episodes))

        for i_episode in range(1, episodes + 1):
            # Generate an episode.
            # An episode is an array of (state, action, reward) tuples
            episode = []
            s = env.reset()
            for t in range(time_steps):
                a = self.get_action(s)
                ns, r, d, _ = env.step(a)

                stats.episode_rewards[i_episode-1] += r
                stats.episode_lengths[i_episode-1] = t

                episode.append((s, a, r))

                if d:
                    break
                s = ns

            for t in range(len(episode)):
                # Find the first occurance of the state in the episode
                s, a, r = episode[t]
                # get sample return
                r_remains = np.array([r for _, _, r in episode[t:]])
                r_remains[-1] = 0
                w_remains = (self._gamma ** np.array(list(range(len(episode) - t))))

                if self._use_baseline:
                    self._V_optimizer.zero_grad()
                    delta = tt(np.sum(r_remains * w_remains)) - self._V(tt(s)).detach()
                else:
                    delta = tt(np.sum(r_remains * w_remains)).detach()
                self._pi_optimizer.zero_grad()

                act_prob = self._pi(tt(s))[a]  # returns softmax value of action probs
                out_pi = -self._step_size * (self._gamma**t) * delta * torch.log(act_prob)
                if self._use_baseline:
                    out_v = -self._step_size_v * delta * self._V(tt(s))
                    out_v.backward()
                    self._V_optimizer.step()
                out_pi.backward()
                self._pi_optimizer.step()

            print("\r{} Steps in Episode {}/{}. Reward {}".format(len(episode),
                  i_episode, episodes, sum([e[2] for i, e in enumerate(episode)])))

        return stats


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    fig1.savefig('episode_lengths.png')
    if noshow:
        plt.close(fig1)
    else:
        plt.show()

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(
        smoothing_window))
    fig2.savefig('reward.png')
    if noshow:
        plt.close(fig2)
    else:
        plt.show()


if __name__ == "__main__":
    env = MountainCarEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # reinforce = REINFORCE(state_dim, action_dim, gamma=0.99, step_size=1, use_baseline=False)
    reinforce = REINFORCE(state_dim, action_dim, gamma=0.99, step_size=1, use_baseline=True, step_size_v=0.1)

    episodes = 1000
    time_steps = 500

    stats = reinforce.train(episodes, time_steps)

    plot_episode_stats(stats)

    for _ in range(5):
        s = env.reset()
        for _ in range(500):
            env.render()
            a = reinforce.get_action(s)
            s, _, d, _ = env.step(a)
            if d:
                break
