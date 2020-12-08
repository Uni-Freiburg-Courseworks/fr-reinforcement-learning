import numpy as np
from gridworld import q_func, Actions


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
      env: OpenAI environment. env.P represents the transition probabilities of the environment.
      theta: Stopping threshold. If the value of all states changes less than theta
        in one iteration we are done.
      discount_factor: lambda time discount factor.

    Returns:
      A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])
    n_evaluations = 0

    while True:
        max_error = 0.0

        for state, value in enumerate(V):
            action_returns = [q_func(env, state, action, V, discount_factor) for action in Actions]

            best_idx = np.argmax(action_returns)
            best_action = Actions[best_idx]
            best_return = action_returns[best_idx]
            max_error = max(max_error, abs(best_return - V[state]))
            V[state] = best_return

            policy[state] = np.zeros(env.nA)
            policy[state][best_action] = 1.0

            n_evaluations += 1

        if max_error < theta:
            break


    print(f"value iteration: {n_evaluations} evaluations")
    return policy, V
