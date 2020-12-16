import sys
import numpy as np
from collections import defaultdict, namedtuple
from tqdm import tqdm

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
        # TODO: 1. Implement this!
        q_vals = Q[observation]
        best_action = np.argmax(q_vals)
        prob = np.ones(nA) * epsilon / nA
        prob[best_action] += 1 - epsilon

        return prob

    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
      env: environment.
      num_episodes: Number of episodes to run for.
      discount_factor: Lambda time discount factor.
      alpha: TD learning rate.
      epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
      A tuple (Q, episode_lengths).
      Q is the optimal action-value function, a dictionary mapping state -> action values.
      stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.nA))

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

        done = False
        s = env.reset()
        episode_length = 0
        episode_reward = 0
        while not done:
            a = np.random.choice(env.nA, p=policy(s))
            s1, reward, done, _ = env.step(a)
            q1_max = np.max(Q[s1])
            Q[s] += alpha * (reward + discount_factor * q1_max - Q[s])
            s = s1
            episode_length += 1
            episode_reward += reward

        stats.episode_lengths[i_episode] = episode_length
        stats.episode_rewards[i_episode] = episode_reward

    return Q, stats
