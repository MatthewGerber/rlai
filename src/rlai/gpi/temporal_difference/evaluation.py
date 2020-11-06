import enum
from typing import Dict, Set, Tuple, Optional

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.gpi.utils import lazy_initialize_q_S_A, initialize_q_S_A
from rlai.meta import rl_text
from rlai.states.mdp import MdpState
from rlai.utils import IncrementalSampleAverager, sample_list_item


@rl_text(chapter=6, page=130)
class Mode(enum.Enum):
    """
    Evaluation modes for temporal-difference evaluation:  SARSA (on-policy, Q-Learning (off-policy), and Expected SARSA
    (off-policy).
    """

    # On-policy SARSA.
    SARSA = enum.auto()

    # Off-policy Q-learning:  The agent policy is used to generate episodes, and it can be any epsilon-soft policy. The
    # target policy is arranged to be greedy with respect to the estimated q-values (i.e., it is the optimal policy). As
    # a result, the Q-values converge to those of the optimal policy.
    Q_LEARNING = enum.auto()

    # Off-policy expected SARSA.
    EXPECTED_SARSA = enum.auto()


@rl_text(chapter=6, page=130)
def evaluate_q_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        num_episodes: int,
        alpha: Optional[float],
        mode: Mode,
        initial_q_S_A: Dict[MdpState, Dict[Action, IncrementalSampleAverager]] = None
) -> Tuple[Dict[MdpState, Dict[Action, IncrementalSampleAverager]], Set[MdpState], float]:
    """
    Perform temporal-difference (TD) evaluation of an agent's policy within an environment, returning state-action
    values. This evaluation function implements both on-policy TD learning (SARSA) as well as off-policy TD learning
    (Q-learning and expected SARSA).

    :param agent: Agent containing target policy to be optimized.
    :param environment: Environment.
    :param num_episodes: Number of episodes to execute.
    :param alpha: Constant step size to use when updating Q-values, or None for 1/n step size.
    :param mode: Evaluation mode (see `rlai.gpi.temporal_difference.evaluation.Mode`).
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

        # simulate until episode termination. begin by taking an action in the first state.
        curr_t = 0
        curr_a = agent.act(curr_t)
        total_reward = 0.0
        while not curr_state.terminal:

            next_state, next_t, next_reward = curr_state.advance(environment, curr_t, curr_a)
            agent.sense(next_state, next_t)

            next_a = None

            # if the next state is terminal, then all next q-values are zero.
            if next_state.terminal:
                next_state_q_s_a = 0.0
            else:

                # EXPECTED_SARSA:  get expected q-value based on current policy and q-value estimates
                if mode == Mode.EXPECTED_SARSA:
                    next_state_q_s_a = sum(
                        (agent.pi[next_state][a] if next_state in agent.pi else 1 / len(next_state.AA)) * (q_S_A[next_state][a].get_value() if next_state in q_S_A and a in q_S_A[next_state] else 0.0)
                        for a in next_state.AA
                    )
                else:

                    # SARSA:  agent determines the t-d target action as well as the episode's next action (on-policy)
                    if mode == Mode.SARSA:
                        td_target_a = next_a = agent.act(next_t)

                    # Q-LEARNING:  select the action with max q-value from the next state. if no q-values are estimated,
                    # then select the action uniformly randomly.
                    elif mode == Mode.Q_LEARNING:
                        if next_state in q_S_A and len(q_S_A[next_state]) > 0:
                            td_target_a = max(q_S_A[next_state], key=lambda a: q_S_A[next_state][a].get_value())
                        else:
                            td_target_a = sample_list_item(next_state.AA, probs=None, random_state=environment.random_state)
                    else:
                        raise ValueError(f'Unknown TD mode:  {mode}')

                    # get the next state-action value if we have an estimate for it; otherwise, it's zero.
                    if next_state in q_S_A and td_target_a in q_S_A[next_state]:
                        next_state_q_s_a = q_S_A[next_state][td_target_a].get_value()
                    else:
                        next_state_q_s_a = 0.0

            # t-d target:  actual realized reward of the current state-action pair, plus the (bootstrapped) discounted
            # future value of the next state-action value, as estimated by one of the modes.
            td_target = next_reward.r + agent.gamma * next_state_q_s_a

            # update the value of the current state/action pair with the t-d target
            lazy_initialize_q_S_A(q_S_A=q_S_A, state=curr_state, a=curr_a, alpha=alpha, weighted=False)
            q_S_A[curr_state][curr_a].update(td_target)
            evaluated_states.add(curr_state)

            # advance the episode
            curr_t = next_t
            curr_state = next_state
            curr_a = agent.act(next_t) if next_a is None else next_a  # if the next action has not been set, then we're off-policy and the agent should determine the next action.
            total_reward += next_reward.r

        episode_reward_averager.update(total_reward)

        episodes_finished = episode_i + 1
        if episodes_finished % episodes_per_print == 0:
            print(f'Finished {episodes_finished} of {num_episodes} episode(s).')

    return q_S_A, evaluated_states, episode_reward_averager.get_value()
