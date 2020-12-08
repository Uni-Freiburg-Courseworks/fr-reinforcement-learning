import numpy as np
from gridworld import UP, DOWN, LEFT, RIGHT, Actions, q_func
import math


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
      policy: [S, A] shaped matrix representing the policy.
      env: OpenAI env. env.P represents the transition probabilities of the environment.
        env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
      theta: We stop evaluation once our value function change is less than theta for all states.
      discount_factor: gamma discount factor.

    Returns:
      Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    n_evaluations = 0
    while True:
        max_error = 0.0
        V_new = np.zeros_like(V)
        for state, value in enumerate(V):
            value_new = 0.0
            for action in [UP, DOWN, LEFT, RIGHT]:
                pol_prob = policy[state][action]
                value_new += pol_prob * q_func(env, state, action, V, discount_factor)
            max_error = max(max_error, abs(value_new - value))
            V_new[state] = value_new
        V = V_new
        n_evaluations += 1
        if max_error < theta:
            break

    print(f"policy evaluation: {n_evaluations} evaluations")
    return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
      env: The OpenAI envrionment.
      policy_eval_fn: Policy Evaluation function that takes 3 arguments:
        policy, env, discount_factor.
      discount_factor: Lambda discount factor.

    Returns:
      A tuple (policy, V).
      policy is the optimal policy, a matrix of shape [S, A] where each state s
      contains a valid probability distribution over actions.
      V is the value function for the optimal policy.

    """
    V = np.zeros(env.nS)
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        policy_stable = True
        V = policy_eval_fn(policy, env, discount_factor=discount_factor)

        for state, value in enumerate(V):
            q_values = [q_func(env, state, action, V, discount_factor) for action in Actions]
            grd_idx = np.argmax(q_values)
            grd_act = Actions[grd_idx]
            grd_policy = np.zeros(env.nA)
            grd_policy[grd_act] = 1.0
            if (grd_policy != policy[state]).any():
                policy_stable = False
                policy[state] = grd_policy

        if policy_stable:
            break

    return policy, V
