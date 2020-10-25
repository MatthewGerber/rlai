import pickle
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


def resume_iterate_value_q_pi_from_checkpoint(
        checkpoint_path: str,
        new_checkpoint_path: Optional[str] = None,
        **new_args
):
    if new_checkpoint_path is None:
        new_checkpoint_path = checkpoint_path

    with open(checkpoint_path, 'rb') as checkpoint_file:
        resume_args = pickle.load(checkpoint_file)

    resume_args['checkpoint_path'] = new_checkpoint_path

    if new_args is not None:
        resume_args.update(new_args)

    iterate_value_q_pi(**resume_args)


@rl_text(chapter=5, page=99)
def iterate_value_q_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        num_improvements: int,
        num_episodes_per_improvement: int,
        epsilon: float,
        num_improvements_per_plot: Optional[int] = None,
        num_improvements_per_checkpoint: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        initial_q_S_A: Optional[Dict] = None
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
    :param num_improvements_per_checkpoint: Number of improvements per checkpoint save.
    :param checkpoint_path: Checkpoint path. Must be provided if `num_improvements_per_checkpoint` is provided.
    :param initial_q_S_A: Initial state-action value estimates (primarily useful for restarting from a checkpoint).
    :return: Final state-action value estimates.
    """

    if epsilon == 0.0:
        warnings.warn('Epsilon is 0.0. Exploration and convergence not guaranteed. Consider passing a value > 0 to maintain exploration.')

    q_S_A = initial_q_S_A
    i = 0
    per_episode_average_rewards = []
    state_space_size = []
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
        state_space_size.append(len(q_S_A))

        i += 1

        if num_improvements_per_plot is not None and i % num_improvements_per_plot == 0:
            plt.close('all')
            plt.plot(per_episode_average_rewards, '-', label='win-loss')
            plt.xlabel('Time step')
            plt.ylabel('Win-loss differential (% win - % loss)')
            plt.grid()
            state_space_ax = plt.twinx()
            state_space_ax.plot(state_space_size, '--', label='# states')
            state_space_ax.set_ylabel('# states')
            state_space_ax.legend()
            plt.show()

        if num_improvements_per_checkpoint is not None and i % num_improvements_per_checkpoint == 0:

            resume_args = {
                'agent': agent,
                'environment': environment,
                'num_improvements': num_improvements,
                'num_episodes_per_improvement': num_episodes_per_improvement,
                'epsilon': epsilon,
                'num_improvements_per_plot': num_improvements_per_plot,
                'num_improvements_per_checkpoint': num_improvements_per_checkpoint,
                'initial_q_S_A': q_S_A
            }

            with open(checkpoint_path, 'wb') as checkpoint_file:
                pickle.dump(resume_args, checkpoint_file)

        if i >= num_improvements:
            break

    print(f'Value iteration of q_pi terminated after {i} iteration(s).')

    return q_pi
