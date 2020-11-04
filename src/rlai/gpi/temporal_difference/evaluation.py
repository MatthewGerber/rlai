from typing import Dict, Set, Tuple, Optional

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.gpi.utils import lazy_initialize_q_S_A, initialize_q_S_A
from rlai.meta import rl_text
from rlai.states.mdp import MdpState
from rlai.utils import IncrementalSampleAverager, sample_list_item


@rl_text(chapter=6, page=130)
def evaluate_q_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        num_episodes: int,
        alpha: Optional[float],
        q_learning: bool,
        initial_q_S_A: Dict[MdpState, Dict[Action, IncrementalSampleAverager]] = None
) -> Tuple[Dict[MdpState, Dict[Action, IncrementalSampleAverager]], Set[MdpState], float]:
    """
    Perform temporal-difference evaluation of an agent's policy within an environment, returning state-action values.

    :param agent: Agent containing target policy to be optimized.
    :param environment: Environment.
    :param num_episodes: Number of episodes to execute.
    :param alpha: Constant step size to use when updating Q-values, or None for 1/n step size.
    :param q_learning: True to perform off-policy Q-learning. In off-policy Q-learning, the `agent` policy is used to
    generate episodes, and it can be any epsilon-soft policy. The target policy is arranged to be greedy with respect to
    the estimated q-values (i.e., it is the optimal policy). As a result, the Q-values converge to those of the optimal
    policy.
    :param initial_q_S_A: Initial guess at state-action value, or None for no guess.
    :return: 3-tuple of (1) dictionary of all MDP states and their action-value averagers under the agent's policy, (2)
    set of only those states that were evaluated, and (3) the average reward obtained per episode.
    """

    print(f'Running temporal-difference evaluation of q_pi for {num_episodes} episode(s).')

    evaluated_states = set()

    q_S_A = initialize_q_S_A(initial_q_S_A, environment, evaluated_states)

    episode_reward_averager = IncrementalSampleAverager()
    episodes_per_print = max(1, int(num_episodes * 0.05))
    for episode_i in range(num_episodes):

        # reset the environment for the new run, and reset the agent accordingly.
        curr_state = environment.reset_for_new_run()
        agent.reset_for_new_run(curr_state)

        # simulate until episode termination. begin by taking an action in the current state.
        curr_t = 0
        curr_a = agent.act(curr_t)
        total_reward = 0.0
        while not curr_state.terminal:

            next_state, next_t, next_reward = curr_state.advance(environment, curr_t, curr_a)
            agent.sense(next_state, next_t)

            # if the next state is terminal, then it might not have any feasible actions for the agent to select from.
            # if this is the case, then use a dummy action to allow the update to go through. the corresponding state-
            # action value will always be zero below.
            if next_state.terminal and len(next_state.AA) == 0:
                td_target_a = Action(-1)

            # under q-learning, the action used to form the t-d target is selected to maximize the q-value from the
            # next state (if any q-values are estimated). if no q-values are estimated, then select the action randomly.
            elif q_learning:
                if next_state in q_S_A and len(q_S_A[next_state]) > 0:
                    td_target_a = max(q_S_A[next_state], key=lambda a: q_S_A[next_state][a].get_value())
                else:
                    td_target_a = sample_list_item(next_state.AA, probs=None, random_state=environment.random_state)

            # under non-q-learning temporal differencing, the agent determines the t-d target action.
            else:
                td_target_a = agent.act(next_t)

            # t-d target:  actual realized reward of the current state-action pair, plus the (bootstrapped) discounted
            # future value of the next state-action pair.
            lazy_initialize_q_S_A(q_S_A=q_S_A, state=next_state, a=td_target_a, alpha=alpha, weighted=False)
            td_target = next_reward.r + agent.gamma * q_S_A[next_state][td_target_a].get_value()

            # update the value of the current state/action pair with the t-d target
            lazy_initialize_q_S_A(q_S_A=q_S_A, state=curr_state, a=curr_a, alpha=alpha, weighted=False)
            q_S_A[curr_state][curr_a].update(td_target)
            evaluated_states.add(curr_state)

            # advance the episode
            curr_t = next_t
            curr_state = next_state
            curr_a = agent.act(next_t) if q_learning else td_target_a  # get the next action from the agent if we're q-learning; otherwise, if not q-learning, then we already have the next action as the t-d target action.
            total_reward += next_reward.r

        episode_reward_averager.update(total_reward)

        episodes_finished = episode_i + 1
        if episodes_finished % episodes_per_print == 0:
            print(f'Finished {episodes_finished} of {num_episodes} episode(s).')

    return q_S_A, evaluated_states, episode_reward_averager.get_value()
