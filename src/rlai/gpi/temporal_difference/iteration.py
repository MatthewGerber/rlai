import os
import pickle
from datetime import datetime
from typing import Optional, Union

from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment, MdpPlanningEnvironment, PrioritizedSweepingMdpPlanningEnvironment
from rlai.environments.openai_gym import Gym
from rlai.gpi.temporal_difference.evaluation import evaluate_q_pi, Mode
from rlai.gpi.utils import plot_policy_iteration
from rlai.meta import rl_text
from rlai.value_estimation import StateActionValueEstimator


@rl_text(chapter=6, page=130)
def iterate_value_q_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        num_improvements: int,
        num_episodes_per_improvement: int,
        num_updates_per_improvement: Optional[int],
        alpha: Optional[float],
        mode: Union[Mode, str],
        n_steps: Optional[int],
        epsilon: float,
        planning_environment: Optional[MdpPlanningEnvironment],
        make_final_policy_greedy: bool,
        q_S_A: StateActionValueEstimator,
        num_improvements_per_plot: Optional[int] = None,
        num_improvements_per_checkpoint: Optional[int] = None,
        checkpoint_path: Optional[str] = None
):
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
    :param epsilon: Total probability mass to spread across all actions, resulting in an epsilon-greedy policy. Must
    be strictly > 0.
    :param planning_environment: Planning environment to learn and use.
    :param make_final_policy_greedy: Whether or not to make the agent's final policy greedy with respect to the q-values
    that have been learned, regardless of the value of epsilon used to estimate the q-values.
    :param q_S_A: State-action value estimator.
    :param num_improvements_per_plot: Number of improvements to make before plotting the per-improvement average. Pass
    None to turn off all plotting.
    :param num_improvements_per_checkpoint: Number of improvements per checkpoint save.
    :param checkpoint_path: Checkpoint path. Must be provided if `num_improvements_per_checkpoint` is provided.
    """

    if epsilon is None or epsilon <= 0:
        raise ValueError('epsilon must be strictly > 0 for TD-learning')

    if checkpoint_path is not None:
        checkpoint_path = os.path.expanduser(checkpoint_path)

    if isinstance(mode, str):
        mode = Mode[mode]

    i = 0
    iteration_average_reward = []
    iteration_total_states = []
    iteration_num_states_improved = []
    elapsed_seconds_average_rewards = {}
    start_datetime = datetime.now()
    while i < num_improvements:

        print(f'Value iteration {i + 1}:  ', end='')

        # interact with the environment and (optionally) build a model of the environment for planning purposes
        evaluated_states, average_reward = evaluate_q_pi(
            agent=agent,
            environment=environment,
            num_episodes=num_episodes_per_improvement,
            num_updates_per_improvement=num_updates_per_improvement,
            epsilon=epsilon,
            alpha=alpha,
            mode=mode,
            n_steps=n_steps,
            planning_environment=planning_environment,
            q_S_A=q_S_A
        )

        num_states_improved = q_S_A.improve_policy(
            agent=agent,
            states=evaluated_states,
            epsilon=epsilon
        )

        q_S_A.plot(final=False)

        iteration_average_reward.append(average_reward)
        iteration_total_states.append(len(q_S_A))
        iteration_num_states_improved.append(num_states_improved)

        # run planning through a recursive call to the iteration method, passing the planning environment as the
        # environment to interact with and disabling planning in the recursive call.
        if planning_environment is not None:
            print(f'Running {planning_environment.num_planning_improvements_per_direct_improvement} planning improvement(s).')
            iterate_value_q_pi(
                agent=agent,
                environment=planning_environment,
                num_improvements=planning_environment.num_planning_improvements_per_direct_improvement,
                num_episodes_per_improvement=num_episodes_per_improvement,
                num_updates_per_improvement=num_updates_per_improvement,
                alpha=alpha,
                mode=mode,
                n_steps=n_steps,
                epsilon=epsilon,
                planning_environment=None,
                make_final_policy_greedy=False,
                q_S_A=q_S_A,
                num_improvements_per_plot=None,
                num_improvements_per_checkpoint=None,
                checkpoint_path=None
            )
            print('Finished planning.')

        elapsed_seconds = int((datetime.now() - start_datetime).total_seconds())
        if elapsed_seconds not in elapsed_seconds_average_rewards:
            elapsed_seconds_average_rewards[elapsed_seconds] = []

        elapsed_seconds_average_rewards[elapsed_seconds].append(average_reward)

        i += 1

        if num_improvements_per_plot is not None and i % num_improvements_per_plot == 0:
            plot_policy_iteration(iteration_average_reward, iteration_total_states, iteration_num_states_improved, elapsed_seconds_average_rewards)

        if num_improvements_per_checkpoint is not None and i % num_improvements_per_checkpoint == 0:

            # gym environments cannot be pickled, so just save the native id so that we can resume it later.
            gym_native = None
            if isinstance(environment, Gym):
                gym_native = environment.gym_native
                environment.gym_native = environment.gym_native.spec.id

            # priority queues cannot be pickled since they contain thread locks
            if isinstance(planning_environment, PrioritizedSweepingMdpPlanningEnvironment):
                planning_environment.state_action_priority = None

            resume_args = {
                'agent': agent,
                'environment': environment,
                'num_improvements': num_improvements - i,
                'num_episodes_per_improvement': num_episodes_per_improvement,
                'num_updates_per_improvement': num_updates_per_improvement,
                'alpha': alpha,
                'mode': mode,
                'n_steps': n_steps,
                'epsilon': epsilon,
                'planning_environment': planning_environment,
                'make_final_policy_greedy': make_final_policy_greedy,
                'q_S_A': q_S_A,
                'num_improvements_per_plot': num_improvements_per_plot,
                'num_improvements_per_checkpoint': num_improvements_per_checkpoint,
                'checkpoint_path': checkpoint_path
            }

            with open(checkpoint_path, 'wb') as checkpoint_file:
                pickle.dump(resume_args, checkpoint_file)

            if gym_native is not None:
                environment.gym_native = gym_native

    print(f'Value iteration of q_pi terminated after {i} iteration(s).')

    q_S_A.plot(final=True)

    if make_final_policy_greedy:
        q_S_A.improve_policy(
            agent=agent,
            states=None,
            epsilon=0.0
        )
