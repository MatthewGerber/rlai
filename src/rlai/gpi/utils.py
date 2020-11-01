import pickle
from typing import Dict, Optional, Set, List, Callable

import matplotlib.pyplot as plt

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.states.mdp import MdpState
from rlai.utils import IncrementalSampleAverager


def initialize_q_S_A(
        initial_q_S_A: Dict[MdpState, Dict[Action, IncrementalSampleAverager]],
        environment: MdpEnvironment,
        evaluated_states: Set[MdpState]
) -> Dict[MdpState, Dict[Action, IncrementalSampleAverager]]:
    """
    Initialize q_S_A structure.

    :param initial_q_S_A: Initial guess at state-action value, or None for no guess.
    :param environment: Environment.
    :param evaluated_states: Evaluated states.
    :return: q_S_A structure.
    """

    # if no initial guess is provided, then start with an averager for each terminal state. these should never be used.
    if initial_q_S_A is None:
        q_S_A = {
            terminal_state: {
                a: IncrementalSampleAverager()
                for a in terminal_state.AA
            }
            for terminal_state in environment.terminal_states
        }

        for s in q_S_A:
            evaluated_states.add(s)

    # set to initial guess
    else:
        q_S_A = initial_q_S_A

    return q_S_A


def lazy_initialize_q_S_A(
        q_S_A: Dict[MdpState, Dict[Action, IncrementalSampleAverager]],
        state: MdpState,
        a: Action,
        alpha: Optional[float],
        weighted: bool
):
    """
    Lazy-initialize the reward averager for a state-action pair.

    :param q_S_A: State-action averagers.
    :param state: State.
    :param a: Action.
    :param alpha: Step size.
    :param weighted: Whether the averger should be weighted.
    """

    if state not in q_S_A:
        q_S_A[state] = {}

    if a not in q_S_A[state]:
        q_S_A[state][a] = IncrementalSampleAverager(alpha=alpha, weighted=weighted)


def get_q_pi_for_evaluated_states(
        q_S_A: Dict[MdpState, Dict[Action, IncrementalSampleAverager]],
        evaluated_states: Set[MdpState]
):
    """
    Get the q_pi that only includes states visited in the current iteration. There is no need to update the agent's
    policy for states that weren't evaluated, and this will dramatically cut down computation for environments with
    large state spaces.

    :param q_S_A:
    :param evaluated_states:
    :return:
    """

    q_pi = {
        s: {
            a: q_S_A[s][a].get_value()
            for a in q_S_A[s]
        }
        for s in q_S_A
        if s in evaluated_states
    }

    return q_pi


def plot_policy_iteration(
        iteration_average_reward: List[float],
        iteration_total_states: List[int],
        iteration_num_states_updated: List[int]
):
    """
    Plot status of policy iteration.

    :param iteration_average_reward: Average reward per iteration.
    :param iteration_total_states: Total number of states per iteration.
    :param iteration_num_states_updated: Number of states updated per iteration.
    """

    plt.close('all')
    plt.plot(iteration_average_reward, '-', label='average')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.grid()
    state_space_ax = plt.twinx()
    state_space_ax.plot(iteration_total_states, '--', color='orange', label='total')
    state_space_ax.plot(iteration_num_states_updated, '-', color='orange', label='updated')
    state_space_ax.set_yscale('log')
    state_space_ax.set_ylabel('# states')
    state_space_ax.legend()
    plt.show()


def resume_from_checkpoint(
        checkpoint_path: str,
        resume_function: Callable,
        new_checkpoint_path: Optional[str] = None,
        resume_args_mutator: Callable = None,
        **new_args
) -> MdpAgent:
    """
    Resume the execution of a previous call to `rlai.gpi.monte_carlo.iteration.iterate_value_q_pi`, based on a stored
    checkpoint.

    :param checkpoint_path: Path to checkpoint file.
    :param resume_function: Function to resume.
    :param new_checkpoint_path: Path to new checkpoint file, if the original should be left as it is. Pass `None` to
    use and overwrite `checkpoint_path` with new checkpoints.
    :param resume_args_mutator: A function called prior to resumption. This function will be passed a dictionary of
    arguments comprising the checkpoint. The passed function can change these arguments if desired.
    :param new_args: As a simpler alternative to `resume_args_mutator`, pass any keyword arguments that should replace
    those in the checkpoint.
    :return: The updated agent.
    """

    if new_checkpoint_path is None:
        new_checkpoint_path = checkpoint_path

    print('Reading checkpoint file to resume...', end='')
    with open(checkpoint_path, 'rb') as checkpoint_file:
        resume_args = pickle.load(checkpoint_file)
    print('.done')

    resume_args['checkpoint_path'] = new_checkpoint_path

    if new_args is not None:
        resume_args.update(new_args)

    if resume_args_mutator is not None:
        resume_args_mutator(**resume_args)

    resume_function(**resume_args)

    return resume_args['agent']
