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
import random
import copy

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def tt(ndarray):
    return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)


def soft_update(target, source, tau) -> None:
    # Note: the target weights are updated by taking 1-tau of themselves and adding tau times the weights of the source weights
    with torch.no_grad():
        source_param = source.named_parameters()
        for k, source_p in source_param:
            target_p = target.state_dict()[k]
            target.state_dict()[k] = (1 - tau) * target_p + tau * source_p


def hard_update(target, source) -> None:
    source_copy = copy.deepcopy(source)
    target.load_state_dict(source_copy.state_dict())


class Q(nn.Module):
    def __init__(self, state_dim, action_dim, non_linearity=F.relu, hidden_dim=50):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self._non_linearity = non_linearity

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)

Transition = namedtuple("Transition", ["states", "actions", "next_states", "rewards", "terminal_flags"])
class ReplayBuffer:
    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, max_size):
        self._data = Transition(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])
        self._size = 0
        self._max_size = max_size

    def add_transition(self, state, action, next_state, reward, done):

        if self._size == self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)
            self._size -= 1

        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._size += 1

    def random_next_batch(self, batch_size) -> Transition:
        """get a minibatch from the replay buffer with bs=`batch_size` with each item being a tensor"""
        assert self._size > 0, "You can only sample when there are already transitions in the buffer!"

        idx = random.choices(list(range(self._size)), k=batch_size)

        s_states = torch.tensor(self._data.states, dtype=torch.float)[idx].reshape(batch_size, -1)
        s_actions = torch.tensor(self._data.actions)[idx].reshape(batch_size)
        s_next_states = torch.tensor(self._data.next_states, dtype=torch.float)[idx].reshape(batch_size, -1)
        s_rewards = torch.tensor(self._data.rewards, dtype=torch.float)[idx].reshape(batch_size)
        s_terminal_flags = torch.tensor(self._data.terminal_flags)[idx].reshape(batch_size)

        return Transition(
            states = s_states,
            actions = s_actions,
            next_states = s_next_states,
            rewards = s_rewards,
            terminal_flags = s_terminal_flags
        )


class DQN:
    def __init__(self, state_dim, action_dim, gamma, bs, update_type='soft', soft_update_tau=0.5, hard_update_cycle=10):
        self._q = Q(state_dim, action_dim)
        self._q_target = Q(state_dim, action_dim)

        # self._q.cuda()
        # self._q_target.cuda()

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
        self._action_dim = action_dim
        self._bs = bs
        self._update_type = update_type
        self._soft_update_tau = soft_update_tau
        self._hard_update_cycle = hard_update_cycle
        self._hard_update_cnt = 0

        self._replay_buffer = ReplayBuffer(1e6)

    def get_action(self, x, epsilon):
        u = np.argmax(self._q(tt(x)).cpu().detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def train(self, episodes, time_steps, epsilon):
        stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes))

        for e in range(episodes):
            print("%s/%s" % (e + 1, episodes))
            s = env.reset()
            for t in range(time_steps):
                a = self.get_action(s, epsilon)
                ns, r, d, _ = env.step(a)

                stats.episode_rewards[e] += r
                stats.episode_lengths[e] = t

                self._replay_buffer.add_transition(s, a, ns, r, d)
                trans = self._replay_buffer.random_next_batch(self._bs)

                s_states = trans.states
                s_actions = trans.actions
                s_next_states = trans.next_states
                s_rewards = trans.rewards

                self._q_optimizer.zero_grad()
                loss = self._loss_function(
                    self._q(s_states)[torch.arange(self._bs), s_actions],
                    s_rewards + self._gamma * self._q_target(s_next_states).max(dim=1)[0]
                )
                loss.backward()
                self._q_optimizer.step()

                # update target Q network
                if self._update_type == 'soft':
                    soft_update(self._q_target, self._q, self._soft_update_tau)
                elif self._update_type == 'hard':
                    self._hard_update_cnt += 1
                    if self._hard_update_cnt % self._hard_update_cycle == 0:
                        hard_update(self._q_target, self._q)
                else:
                    raise ValueError(f"Unsupported update_type={self._update_type}!")

                if d:
                    break

                s = ns

            print(f"e_r={stats.episode_rewards[e]}, e_l={stats.episode_lengths[e]}")

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
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    fig2.savefig('reward.png')
    if noshow:
        plt.close(fig2)
    else:
        plt.show()


if __name__ == "__main__":
    env = MountainCarEnv()  # gym.make("MountainCar-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    dqn = DQN(state_dim, action_dim, gamma=0.99, bs=1, update_type='hard')

    episodes = 100
    time_steps = 200
    epsilon = 0.2

    stats = dqn.train(episodes, time_steps, epsilon)

    plot_episode_stats(stats)

    for _ in range(5):
        s = env.reset()
        for _ in range(200):
            env.render()
            a = dqn.get_action(s, epsilon)
            s, _, d, _ = env.step(a)
            if d:
                break
