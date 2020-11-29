import enum
from typing import Dict, Set, Tuple, Optional

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.gpi.utils import lazy_initialize_q_S_A, initialize_q_S_A
from rlai.meta import rl_text
from rlai.planning.environment_models import EnvironmentModel
from rlai.states.mdp import MdpState
from rlai.utils import IncrementalSampleAverager, sample_list_item


@rl_text(chapter=6, page=130)
class Mode(enum.Enum):
    """
    Modes of temporal-difference evaluation:  SARSA (on-policy), Q-Learning (off-policy), and Expected SARSA
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
        n_steps: Optional[int],
        environment_model: Optional[EnvironmentModel],
        initial_q_S_A: Dict[MdpState, Dict[Action, IncrementalSampleAverager]]
) -> Tuple[Dict[MdpState, Dict[Action, IncrementalSampleAverager]], Set[MdpState], float]:
    """
    Perform temporal-difference (TD) evaluation of an agent's policy within an environment, returning state-action
    values. This evaluation function implements both on-policy TD learning (SARSA) as well as off-policy TD learning
    (Q-learning and expected SARSA), and n-step updates are implemented for all learning modes.

    :param agent: Agent containing target policy to be optimized.
    :param environment: Environment.
    :param num_episodes: Number of episodes to execute.
    :param alpha: Constant step size to use when updating Q-values, or None for 1/n step size.
    :param mode: Evaluation mode (see `rlai.gpi.temporal_difference.evaluation.Mode`).
    :param n_steps: Number of steps to accumulate rewards before updating estimated state-action values. Must be in the
    range [1, inf], or None for infinite step size (Monte Carlo evaluation).
    :param environment_model: Environment model to be updated with experience gained during evaluation, or None to
    ignore the environment model.
    :param initial_q_S_A: Initial guess at state-action value, or None for no guess.
    :return: 3-tuple of (1) dictionary of all MDP states and their action-value averagers under the agent's policy, (2)
    set of only those states that were evaluated, and (3) the average reward obtained per episode.
    """

    if n_steps is not None and n_steps < 1:
        raise ValueError('The value of n_steps must be in range [1, inf], or None.')

    print(f'Running temporal-difference evaluation of q_pi for {num_episodes} episode(s).')

    evaluated_states = set()

    q_S_A = initialize_q_S_A(initial_q_S_A, environment, evaluated_states)
    episode_reward_averager = IncrementalSampleAverager()
    episodes_per_print = max(1, int(num_episodes * 0.05))
    for episode_i in range(num_episodes):

        # reset the environment for the new run, and reset the agent accordingly.
        curr_state = environment.reset_for_new_run(agent)
        agent.reset_for_new_run(curr_state)

        # simulate until episode termination. begin by taking an action in the first state.
        curr_t = 0
        curr_a = agent.act(curr_t)
        total_reward = 0.0
        t_state_a_g: Dict[int, Tuple[MdpState, Action, float]] = {}  # dictionary from time steps to tuples of state, action, and truncated return.
        while not curr_state.terminal and (environment.T is None or curr_t < environment.T):

            next_state, next_reward = curr_state.advance(
                environment=environment,
                t=curr_t,
                a=curr_a,
                agent=agent
            )

            next_t = curr_t + 1
            agent.sense(next_state, next_t)

            # update environment model (for planning)
            if environment_model is not None:
                environment_model.update(curr_state, curr_a, next_state, next_reward)

            # initialize the n-step accumulator at the current time for the current state and action
            t_state_a_g[curr_t] = (curr_state, curr_a, 0.0)

            # get prior time steps for which truncated return g should be updated. if n_steps is None, then get all
            # prior time steps (equivalent to infinite n_steps, or Monte Carlo).
            if n_steps is None:
                prior_t_values = list(t_state_a_g.keys())
            else:
                # in 1-step td, the earliest time step is the current time step; in 2-step, the earliest time step is
                # the prior time step, etc. always update through the current time step.
                earliest_t = max(0, curr_t - n_steps + 1)
                prior_t_values = range(earliest_t, curr_t + 1)

            # pass reward to prior state-action values, discounting based on time step differences (the reward should
            # not be discounted for the current time step).
            for t in prior_t_values:
                state, a, g = t_state_a_g[t]
                discount = agent.gamma ** (curr_t - t)
                t_state_a_g[t] = (state, a, g + discount * next_reward.r)

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

                    # SARSA:  agent determines the t-d target action as well as the episode's next action, which are
                    # the same (on-policy)
                    if mode == Mode.SARSA:
                        td_target_a = next_a = agent.act(next_t)

                    # Q-LEARNING:  select the action with max q-value from the next state. if no q-values are estimated,
                    # then select the action uniformly randomly.
                    elif mode == Mode.Q_LEARNING:
                        if next_state in q_S_A and len(q_S_A[next_state]) > 0:
                            td_target_a = max(q_S_A[next_state], key=lambda action: q_S_A[next_state][action].get_value())
                        else:
                            td_target_a = sample_list_item(next_state.AA, probs=None, random_state=environment.random_state)
                    else:
                        raise ValueError(f'Unknown TD mode:  {mode}')

                    # get the next state-action value if we have an estimate for it; otherwise, it's zero.
                    if next_state in q_S_A and td_target_a in q_S_A[next_state]:
                        next_state_q_s_a = q_S_A[next_state][td_target_a].get_value()
                    else:
                        next_state_q_s_a = 0.0

                # if we're off-policy, then we won't yet have a next action. ask the agent for it now.
                if next_a is None:
                    next_a = agent.act(next_t)

            # only update if n_steps is finite (not monte carlo)
            if n_steps is not None:
                update_q_S_A(
                    q_S_A=q_S_A,
                    n_steps=n_steps,
                    curr_t=curr_t,
                    t_state_a_g=t_state_a_g,
                    agent=agent,
                    next_state_q_s_a=next_state_q_s_a,
                    alpha=alpha,
                    evaluated_states=evaluated_states
                )

            # advance the episode
            curr_t = next_t
            curr_state = next_state
            curr_a = next_a

            total_reward += next_reward.r

        # flush out the remaining n-step updates, with all next state-action values being zero.
        flush_n_steps = len(t_state_a_g) + 1
        while len(t_state_a_g):
            update_q_S_A(
                q_S_A=q_S_A,
                n_steps=flush_n_steps,
                curr_t=curr_t,
                t_state_a_g=t_state_a_g,
                agent=agent,
                next_state_q_s_a=0.0,
                alpha=alpha,
                evaluated_states=evaluated_states
            )
            curr_t += 1

        episode_reward_averager.update(total_reward)

        episodes_finished = episode_i + 1
        if episodes_finished % episodes_per_print == 0:
            print(f'Finished {episodes_finished} of {num_episodes} episode(s).')

    return q_S_A, evaluated_states, episode_reward_averager.get_value()


def update_q_S_A(
        q_S_A: Dict[MdpState, Dict[Action, IncrementalSampleAverager]],
        n_steps: Optional[int],
        curr_t: int,
        t_state_a_g: Dict[int, Tuple[MdpState, Action, float]],
        agent: MdpAgent,
        next_state_q_s_a: float,
        alpha: float,
        evaluated_states: Set[MdpState]
):
    """
    Update the value of the n-step state/action pair with the n-step TD target. The n-step TD target is the truncated
    sum of discounted rewards obtained over n steps, plus the (bootstrapped) discounted future value of the next
    state-action value, as estimated by one of the TD modes.

    :param q_S_A: State-action value estimators.
    :param n_steps: Number of time steps to accumulate actual rewards for before updating a state-action value.
    :param curr_t: Current time step.
    :param t_state_a_g: Structore of time, state, action, g accumulators. If an n-step update is feasible at the current
    time step (i.e., if we have accumulated sufficient rewards), then the entry corresponding to the n-step update will
    be deleted from this structure.
    :param agent: Agent.
    :param next_state_q_s_a: Next state-action value.
    :param alpha: Step size.
    :param evaluated_states: Evaluated states.
    """

    # if we're currently far enough along (i.e., have accumulated sufficient rewards), then update.
    update_t = curr_t - n_steps + 1
    if update_t in t_state_a_g:

        update_state, update_a, g = t_state_a_g[update_t]

        # the discount on the next state-action value is exponentiated to n_steps, as we're bootstrapping it starting
        # from the update_t. for n_steps==1, the discount applied to g will have been 1 (agent.gamma**0) and the
        # discount applied here to next_state_q_s_a will be agent.gamma.
        td_target = g + (agent.gamma ** n_steps) * next_state_q_s_a

        # initialize and update the state-action pair with the target
        lazy_initialize_q_S_A(q_S_A=q_S_A, state=update_state, a=update_a, alpha=alpha, weighted=False)
        q_S_A[update_state][update_a].update(td_target)

        # note the evaluated state and remove from our n-step structure
        evaluated_states.add(update_state)
        del t_state_a_g[update_t]
