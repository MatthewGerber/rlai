import logging
import os
import pickle
import warnings
from datetime import datetime
from typing import Optional, List, Dict

from matplotlib.backends.backend_pdf import PdfPages

from rlai.core import MdpAgent
from rlai.core.environments.mdp import MdpEnvironment, MdpPlanningEnvironment
from rlai.docs import rl_text
from rlai.gpi import PolicyImprovementEvent
from rlai.gpi.monte_carlo.evaluation import evaluate_q_pi
from rlai.gpi.state_action_value import ActionValueMdpAgent
from rlai.gpi.utils import plot_policy_iteration
from rlai.utils import RunThreadManager, insert_index_into_path


@rl_text(chapter=5, page=99)
def iterate_value_q_pi(
        agent: ActionValueMdpAgent,
        environment: MdpEnvironment,
        num_improvements: int,
        num_episodes_per_improvement: int,
        update_upon_every_visit: bool,
        planning_environment: Optional[MdpPlanningEnvironment],
        make_final_policy_greedy: bool,
        thread_manager: Optional[RunThreadManager] = None,
        off_policy_agent: Optional[MdpAgent] = None,
        num_improvements_per_plot: Optional[int] = None,
        num_improvements_per_checkpoint: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        pdf_save_path: Optional[str] = None,
        start_improvement: Optional[int] = None
) -> Optional[str]:
    """
    Run Monte Carlo value iteration on an agent using state-action value estimates. This iteration function operates
    over rewards obtained at the end of episodes, so it is only appropriate for episodic tasks.

    :param agent: Agent.
    :param environment: Environment.
    :param num_improvements: Number of policy improvements to make.
    :param num_episodes_per_improvement: Number of policy evaluation episodes to execute for each iteration of
    improvement. Passing `1` will result in the Monte Carlo ES (Exploring Starts) algorithm.
    :param update_upon_every_visit: See `rlai.gpi.monte_carlo.evaluation.evaluate_q_pi`.
    :param planning_environment: Not support. Will raise exception if passed.
    :param make_final_policy_greedy: Whether to make the agent's final policy greedy with respect to the q-values
    that have been learned, regardless of the value of epsilon used to estimate the q-values.
    :param thread_manager: Thread manager. The current function (and the thread running it) will wait on this manager
    before starting each iteration. This provides a mechanism for pausing, resuming, and aborting training. Omit for no
    waiting.
    :param off_policy_agent: See `rlai.gpi.monte_carlo.evaluation.evaluate_q_pi`. The policy of this agent will not
    be updated by this function.
    :param num_improvements_per_plot: Number of improvements to make before plotting the per-improvement average. Pass
    None to turn off all plotting.
    :param num_improvements_per_checkpoint: Number of improvements per checkpoint save.
    :param checkpoint_path: Checkpoint path. Must be provided if `num_improvements_per_checkpoint` is provided.
    :param pdf_save_path: Path where a PDF of all plots is to be saved, or None for no PDF.
    :param start_improvement: 1-based improvement to start at, or None to start at episode 1.
    :return: Final checkpoint path, or None if checkpoints were not saved.
    """

    if thread_manager is None:
        thread_manager = RunThreadManager(True)

    if planning_environment is not None:
        raise ValueError('Planning environments are not currently supported for Monte Carlo iteration.')

    if (agent.q_S_A.epsilon is None or agent.q_S_A.epsilon == 0.0) and off_policy_agent is None:
        warnings.warn(
            'epsilon is 0.0 and there is no off-policy agent. Exploration and convergence not guaranteed. Consider '
            'passing epsilon > 0 or a soft off-policy agent to maintain exploration.'
        )

    if checkpoint_path is not None:
        checkpoint_path = os.path.expanduser(checkpoint_path)

    pdf = None
    if pdf_save_path is not None:
        pdf = PdfPages(os.path.expanduser(pdf_save_path))

    if start_improvement is None:
        improvements_finished = 0
    else:
        improvements_finished = start_improvement - 1
        logging.info(f'Starting with improvement {start_improvement}.')

    iteration_average_reward = []
    iteration_total_states = []
    iteration_num_states_improved = []
    elapsed_seconds_average_rewards: Dict[int, List[float]] = {}
    start_datetime = datetime.now()
    final_checkpoint_path = None
    while improvements_finished < num_improvements:

        thread_manager.wait()
        if thread_manager.abort:
            break

        logging.info(f'Value iteration {improvements_finished + 1}')

        evaluated_states, average_reward = evaluate_q_pi(
            agent=agent,
            environment=environment,
            num_episodes=num_episodes_per_improvement,
            exploring_starts=False,
            update_upon_every_visit=update_upon_every_visit,
            off_policy_agent=off_policy_agent
        )

        num_states_improved = agent.q_S_A.improve_policy(
            agent=agent,
            states=evaluated_states,
            event=PolicyImprovementEvent.FINISHED_EVALUATION
        )

        iteration_average_reward.append(average_reward)
        iteration_total_states.append(len(agent.q_S_A))
        iteration_num_states_improved.append(num_states_improved)

        elapsed_seconds = int((datetime.now() - start_datetime).total_seconds())
        if elapsed_seconds not in elapsed_seconds_average_rewards:
            elapsed_seconds_average_rewards[elapsed_seconds] = []

        elapsed_seconds_average_rewards[elapsed_seconds].append(average_reward)

        improvements_finished += 1

        if num_improvements_per_plot is not None and improvements_finished % num_improvements_per_plot == 0:
            plot_policy_iteration(
                iteration_average_reward,
                iteration_total_states,
                iteration_num_states_improved,
                elapsed_seconds_average_rewards,
                pdf
            )

        if num_improvements_per_checkpoint is not None and improvements_finished % num_improvements_per_checkpoint == 0:

            if checkpoint_path is None:
                raise ValueError('Checkpoint path is required if checkpointing.')

            resume_args = {
                'agent': agent,
                'environment': environment,
                'num_improvements': num_improvements - improvements_finished,
                'num_episodes_per_improvement': num_episodes_per_improvement,
                'update_upon_every_visit': update_upon_every_visit,
                'planning_environment': planning_environment,
                'make_final_policy_greedy': make_final_policy_greedy,
                'off_policy_agent': off_policy_agent,
                'num_improvements_per_plot': num_improvements_per_plot,
                'num_improvements_per_checkpoint': num_improvements_per_checkpoint,
                'checkpoint_path': checkpoint_path,
                'pdf_save_path': pdf_save_path,
                'start_improvement': improvements_finished + 1
            }

            checkpoint_path_with_index = insert_index_into_path(checkpoint_path, improvements_finished)
            final_checkpoint_path = checkpoint_path_with_index
            os.makedirs(os.path.dirname(final_checkpoint_path), exist_ok=True)
            with open(checkpoint_path_with_index, 'wb') as checkpoint_file:
                pickle.dump(resume_args, checkpoint_file)

    logging.info(f'Value iteration of q_pi terminated after {improvements_finished} iteration(s).')

    if make_final_policy_greedy:
        agent.q_S_A.epsilon = 0.0
        agent.q_S_A.improve_policy(
            agent=agent,
            states=None,
            event=PolicyImprovementEvent.MAKING_POLICY_GREEDY
        )

    if pdf is not None:
        pdf.close()
        logging.info(f'PDF of plots is available at {pdf_save_path}')

    return final_checkpoint_path
