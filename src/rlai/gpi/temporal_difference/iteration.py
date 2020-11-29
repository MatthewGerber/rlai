import os
import pickle
from datetime import datetime
from typing import Optional, Dict, Union

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment, MdpPlanningEnvironment
from rlai.environments.openai_gym import Gym
from rlai.gpi.improvement import improve_policy_with_q_pi
from rlai.gpi.temporal_difference.evaluation import evaluate_q_pi, Mode
from rlai.gpi.utils import get_q_pi_for_evaluated_states, plot_policy_iteration
from rlai.meta import rl_text
from rlai.planning.environment_models import StochasticEnvironmentModel
from rlai.states.mdp import MdpState
from rlai.utils import IncrementalSampleAverager


@rl_text(chapter=6, page=130)
def iterate_value_q_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        num_improvements: int,
        num_episodes_per_improvement: int,
        alpha: Optional[float],
        mode: Union[Mode, str],
        n_steps: Optional[int],
        epsilon: float,
        num_planning_improvements_per_direct_improvement: Optional[int],
        make_final_policy_greedy: bool,
        num_improvements_per_plot: Optional[int] = None,
        num_improvements_per_checkpoint: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        initial_q_S_A: Optional[Dict] = None
) -> Dict[MdpState, Dict[Action, IncrementalSampleAverager]]:
    """
    Run temporal-difference value iteration on an agent using state-action value estimates.

    :param agent: Agent.
    :param environment: Environment.
    :param num_improvements: Number of policy improvements to make.
    :param num_episodes_per_improvement: Number of policy evaluation episodes to execute for each iteration of
    improvement.
    :param alpha: Constant step size to use when updating Q-values, or None for 1/n step size.
    :param mode: Evaluation mode (see `rlai.gpi.temporal_difference.evaluation.Mode`).
    :param n_steps: Number of steps (see `rlai.gpi.temporal_difference.evaluation.evaluate_q_pi`).
    :param epsilon: Total probability mass to spread across all actions, resulting in an epsilon-greedy policy. Must
    be strictly > 0.
    :param num_planning_improvements_per_direct_improvement: Number of planning improvements to make for each
    improvement based on actual experience. Pass None for no planning.
    :param make_final_policy_greedy: Whether or not to make the agent's final policy greedy with respect to the q-values
    that have been learned, regardless of the value of epsilon used to estimate the q-values.
    :param num_improvements_per_plot: Number of improvements to make before plotting the per-improvement average. Pass
    None to turn off all plotting.
    :param num_improvements_per_checkpoint: Number of improvements per checkpoint save.
    :param checkpoint_path: Checkpoint path. Must be provided if `num_improvements_per_checkpoint` is provided.
    :param initial_q_S_A: Initial state-action value estimates (primarily useful for restarting from a checkpoint).
    :return: Dictionary of state-action value estimators.
    """

    if epsilon is None or epsilon <= 0:
        raise ValueError('epsilon must be strictly > 0 for TD-learning')

    if checkpoint_path is not None:
        checkpoint_path = os.path.expanduser(checkpoint_path)

    if isinstance(mode, str):
        mode = Mode[mode]

    # initialize a planning environment if needed
    if num_planning_improvements_per_direct_improvement is None:
        planning_environment = None
    else:
        planning_environment = MdpPlanningEnvironment(
            name=f'{environment.name} (planning)',
            random_state=environment.random_state,
            T=None,
            model=StochasticEnvironmentModel()
        )

    q_S_A = initial_q_S_A
    i = 0
    iteration_average_reward = []
    iteration_total_states = []
    iteration_num_states_updated = []
    elapsed_seconds_average_rewards = {}
    start_datetime = datetime.now()
    while i < num_improvements:

        print(f'Value iteration {i + 1}:  ', end='')

        q_S_A, evaluated_states, average_reward = evaluate_q_pi(
            agent=agent,
            environment=environment,
            num_episodes=num_episodes_per_improvement,
            alpha=alpha,
            mode=mode,
            n_steps=n_steps,
            environment_model=None if planning_environment is None else planning_environment.model,
            initial_q_S_A=q_S_A
        )

        q_pi = get_q_pi_for_evaluated_states(q_S_A, evaluated_states)

        num_states_updated = improve_policy_with_q_pi(
            agent=agent,
            q_pi=q_pi,
            epsilon=epsilon
        )

        iteration_average_reward.append(average_reward)
        iteration_total_states.append(len(q_S_A))
        iteration_num_states_updated.append(num_states_updated)

        # run planning by recursively calling into the current function with the planning environment, but without
        # planning, plotting, etc.
        if planning_environment is not None:
            print(f'Running {num_planning_improvements_per_direct_improvement} planning improvement(s).')
            iterate_value_q_pi(
                agent=agent,
                environment=planning_environment,
                num_improvements=num_planning_improvements_per_direct_improvement,
                num_episodes_per_improvement=num_episodes_per_improvement,
                alpha=alpha,
                mode=mode,
                n_steps=n_steps,
                epsilon=epsilon,
                num_planning_improvements_per_direct_improvement=None,
                make_final_policy_greedy=False,
                num_improvements_per_plot=None,
                num_improvements_per_checkpoint=None,
                checkpoint_path=None,
                initial_q_S_A=q_S_A
            )
            print('Finished planning.')

        elapsed_seconds = int((datetime.now() - start_datetime).total_seconds())
        if elapsed_seconds not in elapsed_seconds_average_rewards:
            elapsed_seconds_average_rewards[elapsed_seconds] = []

        elapsed_seconds_average_rewards[elapsed_seconds].append(average_reward)

        i += 1

        if num_improvements_per_plot is not None and i % num_improvements_per_plot == 0:
            plot_policy_iteration(iteration_average_reward, iteration_total_states, iteration_num_states_updated, elapsed_seconds_average_rewards)

        if num_improvements_per_checkpoint is not None and i % num_improvements_per_checkpoint == 0:

            # gym environments cannot be pickled, so just save the native id so that we can resume it later.
            gym_native = None
            if isinstance(environment, Gym):
                gym_native = environment.gym_native
                environment.gym_native = environment.gym_native.spec.id

            resume_args = {
                'agent': agent,
                'environment': environment,
                'num_improvements': num_improvements - i,
                'num_episodes_per_improvement': num_episodes_per_improvement,
                'alpha': alpha,
                'mode': mode,
                'n_steps': n_steps,
                'epsilon': epsilon,
                'num_planning_improvements_per_direct_improvement': num_planning_improvements_per_direct_improvement,
                'make_final_policy_greedy': make_final_policy_greedy,
                'num_improvements_per_plot': num_improvements_per_plot,
                'num_improvements_per_checkpoint': num_improvements_per_checkpoint,
                'initial_q_S_A': q_S_A
            }

            with open(checkpoint_path, 'wb') as checkpoint_file:
                pickle.dump(resume_args, checkpoint_file)

            if gym_native is not None:
                environment.gym_native = gym_native

    print(f'Value iteration of q_pi terminated after {i} iteration(s).')

    if make_final_policy_greedy:
        q_pi = get_q_pi_for_evaluated_states(q_S_A, None)
        improve_policy_with_q_pi(
            agent=agent,
            q_pi=q_pi
        )

    return q_S_A
