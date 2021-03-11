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
from matplotlib import animation
import pandas as pd
from mountain_car import MountainCarEnv
import random
import copy

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def tt(ndarray):
    # to-tensor
    return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)


def soft_update(target, source, tau) -> None:
    # Note: the target weights are updated by taking 1-tau of themselves and adding tau times the weights of the source weights
    # with torch.no_grad():
    #     source_param = source.named_parameters()
    #     for k, source_p in source_param:
    #         target_p = target.state_dict()[k]
    #         target.state_dict()[k] = (1 - tau) * target_p + tau * source_p
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source) -> None:
    # source_copy = copy.deepcopy(source)
    # target.load_state_dict(source_copy.state_dict())
    soft_update(target, source, 1.0)


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

        batch_indices = np.random.choice(len(self._data.states), batch_size)

        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])

        return Transition(
            states = tt(batch_states),
            actions = tt(batch_actions),
            next_states = tt(batch_next_states),
            rewards = tt(batch_rewards),
            terminal_flags = tt(batch_terminal_flags)
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

    def train(self, env, episodes, time_steps, epsilon):
        stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes))

        for e in range(episodes):
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
                s_terminal_flags = trans.terminal_flags

                self._q_optimizer.zero_grad()
                qv_target = s_rewards + (1 - s_terminal_flags.float()) * self._gamma * self._q_target(s_next_states).max(dim=1)[0]
                qv_input = self._q(s_states)[torch.arange(self._bs).long(), s_actions.long()]
                loss = self._loss_function(qv_input, qv_target.detach())
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

            print(f"\r{e+1}/{episodes}: e_r={stats.episode_rewards[e]}, e_l={stats.episode_lengths[e]}", end=' ', flush=True)

        return stats


class DoubleQ:
    def __init__(self, state_dim, action_dim, gamma, bs):
        self._q0 = Q(state_dim, action_dim)
        self._q1 = Q(state_dim, action_dim)

        # self._q.cuda()
        # self._q_target.cuda()

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._q_optimizer0 = optim.Adam(self._q0.parameters(), lr=0.001)
        self._q_optimizer1 = optim.Adam(self._q1.parameters(), lr=0.001)

        self._action_dim = action_dim
        self._bs = bs

        self._replay_buffer = ReplayBuffer(1e6)

    def get_action(self, x, epsilon, net=0):
        if net == 0:
            u = np.argmax(self._q0(tt(x)).cpu().detach().numpy())
        elif net == 1:
            u = np.argmax(self._q1(tt(x)).cpu().detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def train(self, env, episodes, time_steps, epsilon):
        stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes))

        for e in range(episodes):
            print("%s/%s" % (e + 1, episodes))
            s = env.reset()
            for t in range(time_steps):
                net = random.randint(0, 1) # which network to use as current network

                a = self.get_action(s, epsilon, net)
                ns, r, d, _ = env.step(a)

                stats.episode_rewards[e] += r
                stats.episode_lengths[e] = t

                self._replay_buffer.add_transition(s, a, ns, r, d)
                trans = self._replay_buffer.random_next_batch(self._bs)

                s_states = trans.states
                s_actions = trans.actions
                s_next_states = trans.next_states
                s_rewards = trans.rewards
                s_terminal_flags = trans.terminal_flags

                if net == 0:
                    q_optimizer = self._q_optimizer0
                    q_cur = self._q0
                    q_target = self._q1
                else:
                    q_optimizer = self._q_optimizer1
                    q_cur = self._q1
                    q_target = self._q0

                na_best = q_cur(s_next_states).argmax(dim=-1)
                # qv_target = s_rewards + (1 - s_terminal_flags.float()) * self._gamma * q_target(s_next_states).max(dim=1)[0]
                qv_target = s_rewards + (1 - s_terminal_flags.float()) * self._gamma * q_target(s_next_states)[torch.arange(self._bs).long(), na_best.long()]
                qv_input = q_cur(s_states)[torch.arange(self._bs).long(), s_actions.long()]

                q_optimizer.zero_grad()
                loss = self._loss_function(qv_input, qv_target.detach())
                loss.backward()
                q_optimizer.step()

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


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


if __name__ == "__main__":
    env = MountainCarEnv()  # gym.make("MountainCar-v0")
    env.seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # network = DQN(state_dim, action_dim, gamma=0.99, bs=64, update_type='soft', soft_update_tau=0.01)
    # network = DQN(state_dim, action_dim, gamma=0.99, bs=64, update_type='hard', hard_update_cycle=10)
    network = DoubleQ(state_dim, action_dim, gamma=0.99, bs=64)

    episodes = 500
    time_steps = 200
    epsilon = 0.2

    stats = network.train(env, episodes, time_steps, epsilon)

    plot_episode_stats(stats)

    frames = []
    for _ in range(5):
        s = env.reset()
        for _ in range(200):
            f = env.render(mode='rgb_array')
            frames.append(f)
            a = network.get_action(s, epsilon)
            s, _, d, _ = env.step(a)
            if d:
                break

    save_frames_as_gif(frames)
