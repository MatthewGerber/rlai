import logging
import os
import pickle
from datetime import datetime
from typing import Optional, Union, List, Dict

from matplotlib.backends.backend_pdf import PdfPages

from rlai.core.environments.mdp import MdpEnvironment, MdpPlanningEnvironment
from rlai.docs import rl_text
from rlai.gpi import PolicyImprovementEvent
from rlai.gpi.state_action_value import ActionValueMdpAgent
from rlai.gpi.temporal_difference.evaluation import evaluate_q_pi, Mode
from rlai.gpi.utils import plot_policy_iteration
from rlai.utils import RunThreadManager, insert_index_into_path


@rl_text(chapter=6, page=130)
def iterate_value_q_pi(
        agent: ActionValueMdpAgent,
        environment: MdpEnvironment,
        num_improvements: int,
        num_episodes_per_improvement: int,
        num_updates_per_improvement: Optional[int],
        alpha: Optional[float],
        mode: Union[Mode, str],
        n_steps: Optional[int],
        planning_environment: Optional[MdpPlanningEnvironment],
        make_final_policy_greedy: bool,
        thread_manager: Optional[RunThreadManager] = None,
        num_improvements_per_plot: Optional[int] = None,
        num_improvements_per_checkpoint: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        pdf_save_path: Optional[str] = None,
        start_improvement: Optional[int] = None
) -> Optional[str]:
    """
    Run temporal-difference value iteration on an agent using state-action value estimates.

    :param agent: Agent.
    :param environment: Environment.
    :param num_improvements: Number of policy improvements to make.
    :param num_episodes_per_improvement: Number of policy evaluation episodes to execute for each iteration of policy
    improvement.
    :param num_updates_per_improvement: Number of state-action value updates to execute for each iteration of policy
    improvement, or None for policy improvement per specified number of episodes.
    :param alpha: Constant step size to use when updating Q-values, or None for 1/n step size.
    :param mode: Evaluation mode (see `rlai.gpi.temporal_difference.evaluation.Mode`).
    :param n_steps: Number of steps (see `rlai.gpi.temporal_difference.evaluation.evaluate_q_pi`).
    :param planning_environment: Planning environment to learn and use.
    :param make_final_policy_greedy: Whether to make the agent's final policy greedy with respect to the q-values
    that have been learned, regardless of the value of epsilon used to estimate the q-values.
    :param thread_manager: Thread manager. The current function (and the thread running it) will wait on this manager
    before starting each iteration. This provides a mechanism for pausing, resuming, and aborting training. Omit for no
    waiting.
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

    if agent.q_S_A.epsilon is None or agent.q_S_A.epsilon <= 0:
        raise ValueError('epsilon must be strictly > 0 for TD-learning')

    if checkpoint_path is not None:
        checkpoint_path = os.path.expanduser(checkpoint_path)

    if isinstance(mode, str):
        mode = Mode[mode]

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

        # interact with the environment and (optionally) build a model of the environment for planning purposes
        evaluated_states, average_reward = evaluate_q_pi(
            agent=agent,
            environment=environment,
            num_episodes=num_episodes_per_improvement,
            num_updates_per_improvement=num_updates_per_improvement,
            alpha=alpha,
            mode=mode,
            n_steps=n_steps,
            planning_environment=planning_environment
        )

        num_states_improved = agent.q_S_A.improve_policy(
            agent=agent,
            states=evaluated_states,
            event=PolicyImprovementEvent.FINISHED_EVALUATION
        )

        agent.q_S_A.plot(
            final=improvements_finished == num_improvements - 1,
            pdf=pdf
        )

        iteration_average_reward.append(average_reward)
        iteration_total_states.append(len(agent.q_S_A))
        iteration_num_states_improved.append(num_states_improved)

        # run planning through a recursive call to the iteration method, passing the planning environment as the
        # environment to interact with and disabling planning in the recursive call.
        if planning_environment is not None:
            logging.info(
                f'Running {planning_environment.num_planning_improvements_per_direct_improvement} planning '
                'improvement(s).'
            )
            iterate_value_q_pi(
                agent=agent,
                environment=planning_environment,
                num_improvements=planning_environment.num_planning_improvements_per_direct_improvement,
                num_episodes_per_improvement=num_episodes_per_improvement,
                num_updates_per_improvement=num_updates_per_improvement,
                alpha=alpha,
                mode=mode,
                n_steps=n_steps,
                planning_environment=None,
                make_final_policy_greedy=False,
                num_improvements_per_plot=None,
                num_improvements_per_checkpoint=None,
                checkpoint_path=None,
                pdf_save_path=None
            )
            logging.info('Finished planning.')

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
                'num_updates_per_improvement': num_updates_per_improvement,
                'alpha': alpha,
                'mode': mode,
                'n_steps': n_steps,
                'planning_environment': planning_environment,
                'make_final_policy_greedy': make_final_policy_greedy,
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
