import pickle
import warnings
from typing import Dict, Optional, Callable

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
        mutator: Callable = None,
        **new_args
):
    if new_checkpoint_path is None:
        new_checkpoint_path = checkpoint_path

    print('Reading checkpoint file to resume...', end='')
    with open(checkpoint_path, 'rb') as checkpoint_file:
        resume_args = pickle.load(checkpoint_file)
    print('.done')

    resume_args['checkpoint_path'] = new_checkpoint_path

    if new_args is not None:
        resume_args.update(new_args)

    if mutator is not None:
        mutator(**resume_args)

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
    iteration_average_reward = []
    iteration_total_states = []
    iteration_num_states_updated = []
    while True:

        print(f'Value iteration {i + 1}:  ', end='')

        q_S_A, evaluated_states, per_episode_average_reward = evaluate_q_pi(
            agent=agent,
            environment=environment,
            num_episodes=num_episodes_per_improvement,
            exploring_starts=False,
            initial_q_S_A=q_S_A
        )

        # build the q_pi update that only includes states visited in the current iteration. there is no need to update
        # the agent's policy for states that weren't evaluated, and this will dramatically cut down computation for
        # environments with large state spaces.
        q_pi = {
            s: {
                a: q_S_A[s][a].get_value()
                for a in q_S_A[s]
            }
            for s in q_S_A
            if s in evaluated_states
        }

        num_states_updated = improve_policy_with_q_pi(
            agent=agent,
            q_pi=q_pi,
            epsilon=epsilon
        )

        iteration_average_reward.append(per_episode_average_reward)
        iteration_total_states.append(len(q_S_A))
        iteration_num_states_updated.append(num_states_updated)

        i += 1

        if num_improvements_per_plot is not None and i % num_improvements_per_plot == 0:
            plt.close('all')
            plt.plot(iteration_average_reward, '-', label='%win-%loss')
            plt.xlabel('Time step')
            plt.ylabel('Win-loss differential')
            plt.grid()
            state_space_ax = plt.twinx()
            state_space_ax.plot(iteration_total_states, '--', color='orange', label='total')
            state_space_ax.plot(iteration_num_states_updated, '-', color='orange', label='updated')
            state_space_ax.set_yscale('log')
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
