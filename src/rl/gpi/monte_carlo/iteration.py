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
        iterations: int,
        evaluation_episodes_per_improvement: int,
        plot_iterations: bool
) -> Dict[MdpState, Dict[Action, float]]:
    """
    Run value iteration on an agent using state-action value estimates.

    :param agent: Agent.
    :param environment: Environment.
    :param iterations: Total number of iterations to run.
    :param evaluation_episodes_per_improvement: Number of policy evaluation episodes to execute for each iteration
    of improvement. Passing `1` will result in the Monte Carlo ES (Exploring Starts) algorithm.
    :param plot_iterations: Whether or not to plot the per-episode average reward at each iteration.
    :return: Final state-action value estimates.
    """

    q_pi: Optional[Dict[MdpState, Dict[Action, float]]] = None
    i = 0
    per_episode_average_rewards = []
    while True:

        print(f'Value iteration {i + 1}:  ', end='')

        q_pi, per_episode_average_reward = evaluate_q_pi(
            agent=agent,
            environment=environment,
            num_episodes=evaluation_episodes_per_improvement,
            initial_q_S_A=q_pi
        )

        improve_policy_with_q_pi(
            agent=agent,
            q_pi=q_pi
        )

        per_episode_average_rewards.append(per_episode_average_reward)

        i += 1

        if plot_iterations:
            plt.close('all')
            plt.plot(per_episode_average_rewards)
            plt.show()

        if i >= iterations:
            break

    print(f'Value iteration of q_pi terminated after {i} iteration(s).')

    return q_pi
