import warnings
from typing import Dict, Optional

import matplotlib.pyplot as plt

from rl.actions import Action
from rl.agents.mdp import MdpAgent
from rl.environments.mdp import MdpEnvironment
from rl.gpi.improvement import improve_policy_with_q_pi
from rl.gpi.monte_carlo.evaluation import evaluate_q_pi
from rl.meta import rl_text
from rl.states.mdp import MdpState


@rl_text(chapter=5, page=99)
def iterate_value_q_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        num_improvements: int,
        num_episodes_per_improvement: int,
        epsilon: float,
        num_improvements_per_plot: Optional[int] = None
) -> Dict[MdpState, Dict[Action, float]]:
    """
    Run value iteration on an agent using state-action value estimates.

    :param agent: Agent.
    :param environment: Environment.
    :param num_improvements: Number of policy improvements to make.
    :param num_episodes_per_improvement: Number of policy evaluation episodes to execute for each iteration of
    improvement. Passing `1` will result in the Monte Carlo ES (Exploring Starts) algorithm.
    :param epsilon: Total probability mass to spread across all actions, resulting in an epsilon-greedy policy. Must
    be >= 0 if provided.
    :param num_improvements_per_plot: Number of improvements to make before plotting the per-improvement average. Pass
    None to turn off all plotting.
    :return: Final state-action value estimates.
    """

    if epsilon == 0.0:
        warnings.warn('Epsilon is 0.0. Exploration and convergence not guaranteed. Consider passing a value > 0 to maintain exploration.')

    q_S_A = None
    i = 0
    per_episode_average_rewards = []
    while True:

        print(f'Value iteration {i + 1}:  ', end='')

        q_S_A, per_episode_average_reward = evaluate_q_pi(
            agent=agent,
            environment=environment,
            num_episodes=num_episodes_per_improvement,
            exploring_starts=False,
            initial_q_S_A=q_S_A
        )

        q_pi = {
            s: {
                a: q_S_A[s][a].get_value()
                for a in q_S_A[s]
            }
            for s in q_S_A
        }

        improve_policy_with_q_pi(
            agent=agent,
            q_pi=q_pi,
            epsilon=epsilon
        )

        per_episode_average_rewards.append(per_episode_average_reward)

        i += 1

        if num_improvements_per_plot is not None and i % num_improvements_per_plot == 0:
            plt.close('all')
            plt.plot(per_episode_average_rewards)
            plt.show()

        if i >= num_improvements:
            break

    if num_improvements_per_plot is not None:
        plt.close('all')
        plt.plot(per_episode_average_rewards)
        plt.show()

    print(f'Value iteration of q_pi terminated after {i} iteration(s).')

    return q_pi
