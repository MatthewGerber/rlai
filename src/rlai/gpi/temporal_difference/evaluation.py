from typing import Dict, Set, Tuple, Optional

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.gpi.utils import lazy_initialize_q_S_A, initialize_q_S_A
from rlai.meta import rl_text
from rlai.states.mdp import MdpState
from rlai.utils import IncrementalSampleAverager


@rl_text(chapter=6, page=130)
def evaluate_q_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        num_episodes: int,
        alpha: Optional[float],
        initial_q_S_A: Dict[MdpState, Dict[Action, IncrementalSampleAverager]] = None
) -> Tuple[Dict[MdpState, Dict[Action, IncrementalSampleAverager]], Set[MdpState], float]:
    """
    Perform temporal-difference evaluation of an agent's policy within an environment, returning state-action values.

    :param agent: Agent containing target policy to be optimized.
    :param environment: Environment.
    :param num_episodes: Number of episodes to execute.
    :param alpha: Constant step size to use when updating Q-values, or None for 1/n step size.
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

        # simulate until episode termination
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
                next_a = Action(-1)
            else:
                next_a = agent.act(next_t)

            # t-d target:  actual realized reward of the current state-action pair, plus the (bootstrapped) discounted
            # future value of the next state-action pair.
            lazy_initialize_q_S_A(q_S_A=q_S_A, state=next_state, a=next_a, alpha=alpha, weighted=False)
            td_target = next_reward.r + agent.gamma * q_S_A[next_state][next_a].get_value()

            # update the value of the current state/action pair with the t-d target
            lazy_initialize_q_S_A(q_S_A=q_S_A, state=curr_state, a=curr_a, alpha=alpha, weighted=False)
            q_S_A[curr_state][curr_a].update(td_target)
            evaluated_states.add(curr_state)

            # advance
            curr_t = next_t
            curr_state = next_state
            curr_a = next_a
            total_reward += next_reward.r

        episode_reward_averager.update(total_reward)

        episodes_finished = episode_i + 1
        if episodes_finished % episodes_per_print == 0:
            print(f'Finished {episodes_finished} of {num_episodes} episode(s).')

    return q_S_A, evaluated_states, episode_reward_averager.get_value()
