import pickle
from typing import Optional, Dict

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.gpi.improvement import improve_policy_with_q_pi
from rlai.gpi.temporal_difference.evaluation import evaluate_q_pi
from rlai.gpi.utils import get_q_pi_for_evaluated_states, plot_policy_iteration
from rlai.meta import rl_text
from rlai.states.mdp import MdpState


@rl_text(chapter=6, page=130)
def iterate_value_q_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        num_improvements: int,
        num_episodes_per_improvement: int,
        alpha: Optional[float],
        epsilon: Optional[float],
        num_improvements_per_plot: Optional[int] = None,
        num_improvements_per_checkpoint: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        initial_q_S_A: Optional[Dict] = None
) -> Dict[MdpState, Dict[Action, float]]:
    """
    Run temporal-difference value iteration on an agent using state-action value estimates.

    :param agent: Agent.
    :param environment: Environment.
    :param num_improvements: Number of policy improvements to make.
    :param num_episodes_per_improvement: Number of policy evaluation episodes to execute for each iteration of
    improvement.
    :param alpha: Constant step size to use when updating Q-values, or None for 1/n step size.
    :param epsilon: Total probability mass to spread across all actions, resulting in an epsilon-greedy policy. Must
    be >= 0 if provided.
    :param num_improvements_per_plot: Number of improvements to make before plotting the per-improvement average. Pass
    None to turn off all plotting.
    :param num_improvements_per_checkpoint: Number of improvements per checkpoint save.
    :param checkpoint_path: Checkpoint path. Must be provided if `num_improvements_per_checkpoint` is provided.
    :param initial_q_S_A: Initial state-action value estimates (primarily useful for restarting from a checkpoint).
    :return: State-action value estimates from final iteration of improvement.
    """

    q_S_A = initial_q_S_A
    i = 0
    iteration_average_reward = []
    iteration_total_states = []
    iteration_num_states_updated = []
    while True:

        print(f'Value iteration {i + 1}:  ', end='')

        q_S_A, evaluated_states, average_reward = evaluate_q_pi(
            agent=agent,
            environment=environment,
            num_episodes=num_episodes_per_improvement,
            alpha=alpha,
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

        i += 1

        if num_improvements_per_plot is not None and i % num_improvements_per_plot == 0:
            plot_policy_iteration(iteration_average_reward, iteration_total_states, iteration_num_states_updated)

        if num_improvements_per_checkpoint is not None and i % num_improvements_per_checkpoint == 0:

            resume_args = {
                'agent': agent,
                'environment': environment,
                'num_improvements': num_improvements,
                'num_episodes_per_improvement': num_episodes_per_improvement,
                'alpha': alpha,
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
