import sys
import numpy as np
from collections import defaultdict, namedtuple
from tqdm import tqdm
from gridworld import GridworldEnv
import itertools
import random

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
      Q: A dictionary that maps from state -> action-values.
        Each value is a numpy array of length nA (see below)
      epsilon: The probability to select a random action . float between 0 and 1.
      nA: Number of actions in the environment.

    Returns:
      A function that takes the observation as an argument and returns
      the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.random.choice(np.flatnonzero(Q[observation] == Q[observation].max()))
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def dyna_q_learning(env: GridworldEnv, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, n=5):
    """
    Dyna-Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
      env: environment.
      num_episodes: Number of episodes to run for.
      discount_factor: Lambda time discount factor.
      alpha: TD learning rate.
      epsilon: Chance the sample a random action. Float betwen 0 and 1.
      n: number of planning steps

    Returns:
      A tuple (Q, episode_lengths).
      Q is the optimal action-value function, a dictionary mapping state -> action values.
      stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.nA))

    # The model.
    # A nested dictionary that maps state -> (action -> (next state, reward, terminal flag)).
    # M = defaultdict(lambda: np.zeros((env.nA, 3)))
    M = defaultdict(lambda: {})

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.nA)

    for i_episode in tqdm(range(num_episodes)):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # TODO: Implement this!
        state = env.reset()
        done = False
        sum_rewards = 0

        for episode_length in itertools.count():
            act_dist = policy(state)
            act = np.random.choice(env.nA, p=act_dist)
            next_state, reward, done = env.step(act)
            sum_rewards += reward
            Q[state][act] += alpha * (reward + discount_factor * np.max(Q[next_state]) - Q[state][act])
            M[state][act] = (next_state, reward, done)
            state = next_state

            for _ in range(n):
                prev_state = random.choice(list(M.keys()))
                prev_act = random.choice(list(M[prev_state].keys()))
                s1, r, d = M[prev_state][prev_act]
                Q[prev_state][prev_act] += alpha * (r + discount_factor * np.max(Q[s1]) - Q[prev_state][prev_act])

            if done:
                break

        stats.episode_lengths[i_episode] = episode_length
        stats.episode_rewards[i_episode] = sum_rewards

    return Q, stats
