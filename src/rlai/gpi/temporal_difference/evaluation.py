import enum
from functools import partial
from typing import Dict, Set, Tuple, Optional

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment, MdpPlanningEnvironment, PrioritizedSweepingMdpPlanningEnvironment
from rlai.meta import rl_text
from rlai.states.mdp import MdpState
from rlai.utils import IncrementalSampleAverager, sample_list_item
from rlai.value_estimation import StateActionValueEstimator


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
        num_updates_per_improvement: Optional[int],
        epsilon: float,
        alpha: Optional[float],
        mode: Mode,
        n_steps: Optional[int],
        planning_environment: Optional[MdpPlanningEnvironment],
        q_S_A: StateActionValueEstimator
) -> Tuple[Set[MdpState], float]:
    """
    Perform temporal-difference (TD) evaluation of an agent's policy within an environment, returning state-action
    values. This evaluation function implements both on-policy TD learning (SARSA) as well as off-policy TD learning
    (Q-learning and expected SARSA), and n-step updates are implemented for all learning modes.

    :param agent: Agent containing target policy to be optimized.
    :param environment: Environment.
    :param num_episodes: Number of episodes to execute.
    :param num_updates_per_improvement: Number of state-action value updates to execute for each iteration of policy
    improvement, or None for policy improvement per specified number of episodes.
    :param epsilon: Epsilon.
    :param alpha: Constant step size to use when updating Q-values, or None for 1/n step size.
    :param mode: Evaluation mode (see `rlai.gpi.temporal_difference.evaluation.Mode`).
    :param n_steps: Number of steps to accumulate rewards before updating estimated state-action values. Must be in the
    range [1, inf], or None for infinite step size (Monte Carlo evaluation).
    :param planning_environment: Planning environment to learn through experience gained during evaluation, or None to
    not learn an environment model.
    :param q_S_A: State-action value estimator.
    :return: 2-tuple of (1) set of only those states that were evaluated, and (2) the average reward obtained per
    episode.
    """

    if n_steps is not None and n_steps < 1:
        raise ValueError('The value of n_steps must be in range [1, inf], or None.')

    print(f'Running temporal-difference evaluation of q_pi for {num_episodes} episode(s).')

    evaluated_states = set()

    planning = isinstance(environment, MdpPlanningEnvironment)

    # prioritized sampling requires access to the bootstrapped state-action value function, and it also requires
    # access to the state-action value estimators.
    if isinstance(environment, PrioritizedSweepingMdpPlanningEnvironment):
        environment.bootstrap_function = partial(
            get_bootstrapped_state_action_value,
            mode=mode,
            agent=agent,
            q_S_A=q_S_A,
            environment=environment
        )
        environment.q_S_A = q_S_A

    # run episodes
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
        next_state_q_s_a = 0.0
        while not curr_state.terminal and (environment.T is None or curr_t < environment.T):

            advance_result, next_reward = environment.advance(
                state=curr_state,
                t=curr_t,
                a=curr_a,
                agent=agent
            )

            # in the case of a planning-based advancement, the planning environment returns a 3-tuple of the current
            # state, current action, and next state. this is because the planning environment may revise any one of
            # these variables to conduct the planning process (e.g., by prioritized sweeping).
            if planning:
                curr_state, curr_a, next_state = advance_result
            else:
                next_state = advance_result

            next_t = curr_t + 1
            agent.sense(next_state, next_t)

            # if we're building an environment model, then update it with the transition we just observed.
            if planning_environment is not None:
                planning_environment.model.update(curr_state, curr_a, next_state, next_reward)

            # initialize the n-step, truncated return accumulator at the current time for the current state and action
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

            # get the next state's bootstrapped value and next action, based on the bootstrapping mode. note that the
            # bootstrapped next state-action value is only used if we're performing n-step updates.
            next_state_q_s_a, next_a = get_bootstrapped_state_action_value(
                state=next_state,
                t=next_t,
                mode=mode,
                agent=agent,
                q_S_A=q_S_A,
                environment=environment
            )

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
                    evaluated_states=evaluated_states,
                    planning_environment=planning_environment,
                    num_updates_per_improvement=num_updates_per_improvement,
                    epsilon=epsilon
                )

            # advance the episode
            curr_t = next_t
            curr_state = next_state
            curr_a = next_a

            total_reward += next_reward.r

        # flush out the remaining n-step updates. if we terminated because we reached a terminal state, then all next-
        # state values for the updates are zero. if instead we terminated because we reached the maximum number of time
        # steps, then use the bootstrapped next-state value instead. this is an important distinction because these
        # value can be dramatically different, and when discounting is not using (or is very small), the resulting
        # value estimates can be correspondingly different.
        if curr_state.terminal:
            next_state_q_s_a = 0.0

        flush_n_steps = len(t_state_a_g) + 1
        while len(t_state_a_g) > 0:
            update_q_S_A(
                q_S_A=q_S_A,
                n_steps=flush_n_steps,
                curr_t=curr_t,
                t_state_a_g=t_state_a_g,
                agent=agent,
                next_state_q_s_a=next_state_q_s_a,
                alpha=alpha,
                evaluated_states=evaluated_states,
                planning_environment=planning_environment,
                num_updates_per_improvement=num_updates_per_improvement,
                epsilon=epsilon
            )
            curr_t += 1

        episode_reward_averager.update(total_reward)

        episodes_finished = episode_i + 1
        if episodes_finished % episodes_per_print == 0:
            print(f'Finished {episodes_finished} of {num_episodes} episode(s).')

    return evaluated_states, episode_reward_averager.get_value()


def get_bootstrapped_state_action_value(
        state: MdpState,
        t: int,
        mode: Mode,
        agent: MdpAgent,
        q_S_A: StateActionValueEstimator,
        environment: MdpEnvironment
) -> Tuple[float, Action]:
    """
    Get the bootstrapped state-action value for a state, also returning the next action.

    :param state: State.
    :param t: Time step.
    :param mode: Bootstrap mode.
    :param agent: Agent.
    :param q_S_A: Current state-action value estimates.
    :param environment: Environment.
    :return: 2-tuple of the state's bootstrapped state-action value and the next action.
    """

    next_a = None

    # if the state is terminal, then all q-values are zero.
    if state.terminal:
        bootstrapped_s_a_value = 0.0
    else:

        # EXPECTED_SARSA:  get expected q-value based on current policy and q-value estimates
        if mode == Mode.EXPECTED_SARSA:
            bootstrapped_s_a_value = sum(
                (agent.pi[state][a] if state in agent.pi else 1 / len(state.AA)) * (q_S_A[state][a].get_value() if state in q_S_A and a in q_S_A[state] else 0.0)
                for a in state.AA
            )
        else:

            # SARSA:  agent determines the t-d target action as well as the episode's next action, which are the same
            # (we're on-policy)
            if mode == Mode.SARSA:
                td_target_a = next_a = agent.act(t)

            # Q-LEARNING:  select the action with max q-value from the state. if no q-values are estimated, then select
            # the action uniformly randomly.
            elif mode == Mode.Q_LEARNING:
                if state in q_S_A and len(q_S_A[state]) > 0:
                    td_target_a = max(q_S_A[state], key=lambda action: q_S_A[state][action].get_value())
                else:
                    td_target_a = sample_list_item(state.AA, probs=None, random_state=environment.random_state)
            else:  # pragma no cover
                raise ValueError(f'Unknown TD mode:  {mode}')

            # get the state-action value if we have an estimate for it; otherwise, it's zero.
            if state in q_S_A and td_target_a in q_S_A[state]:
                bootstrapped_s_a_value = q_S_A[state][td_target_a].get_value()
            else:
                bootstrapped_s_a_value = 0.0

        # if we're off-policy, then we won't yet have a next action. ask the agent for it now.
        if next_a is None:
            next_a = agent.act(t)

    return bootstrapped_s_a_value, next_a


def update_q_S_A(
        q_S_A: StateActionValueEstimator,
        n_steps: Optional[int],
        curr_t: int,
        t_state_a_g: Dict[int, Tuple[MdpState, Action, float]],
        agent: MdpAgent,
        next_state_q_s_a: float,
        alpha: float,
        evaluated_states: Set[MdpState],
        planning_environment: Optional[MdpPlanningEnvironment],
        num_updates_per_improvement: Optional[int],
        epsilon: float
):
    """
    Update the value of the n-step state/action pair with the n-step TD target. The n-step TD target is the truncated
    sum of discounted rewards obtained over n steps, plus the (bootstrapped) discounted future value of the next
    state-action value, as estimated by one of the TD modes.

    :param q_S_A: State-action value estimator.
    :param n_steps: Number of time steps to accumulate actual rewards for before updating a state-action value.
    :param curr_t: Current time step.
    :param t_state_a_g: Structore of time, state, action, g accumulators. If an n-step update is feasible at the current
    time step (i.e., if we have accumulated sufficient rewards), then the entry corresponding to the n-step update will
    be deleted from this structure.
    :param agent: Agent.
    :param next_state_q_s_a: Next state-action value.
    :param alpha: Step size.
    :param evaluated_states: Evaluated states.
    :param planning_environment: Planning environment to be updated with experience gained during evaluation, or None to
    ignore the environment model.
    :param num_updates_per_improvement: Number of state-action value updates to execute for each iteration of policy
    improvement, or None for policy improvement per specified number of episodes.
    :param epsilon: Epsilon.
    """

    # if we're currently far enough along (i.e., have accumulated sufficient rewards), then update.
    update_t = curr_t - n_steps + 1
    if update_t in t_state_a_g:

        update_state, update_a, g = t_state_a_g[update_t]

        # the discount on the next state-action value is exponentiated to n_steps, as we're bootstrapping it starting
        # from the update_t. for n_steps==1, the discount applied to g will have been 1 (agent.gamma**0) and the
        # discount applied here to next_state_q_s_a will be agent.gamma.
        td_target = g + (agent.gamma ** n_steps) * next_state_q_s_a

        # initialize/get the value estimator for the state-action pair
        q_S_A.initialize(state=update_state, a=update_a, alpha=alpha, weighted=False)
        value_estimator = q_S_A[update_state][update_a]

        # if we're using prioritized planning, then calculate the update error before we update the value estimation.
        prioritized_planning = isinstance(planning_environment, PrioritizedSweepingMdpPlanningEnvironment)
        if prioritized_planning:
            error = td_target - value_estimator.get_value()
        else:
            error = None

        # update the value estimator
        value_estimator.update(td_target)

        # if we're using prioritized-sweep planning, then update the priority queue. note that the priority queue
        # returns values with the lowest priority first. so negate the error to get the state-action pairs with highest
        # error to come out of the queue first.
        if prioritized_planning:
            planning_environment.add_state_action_priority(update_state, update_a, -abs(error))

        # note the evaluated state if it has an index. states will only have indices if they are enumerated up front, or
        # if they are created dynamically in a tabular setting (either discrete or discretized continuous), where we
        # keep track of their identifiers. if they are created dynamically in a function approximation setting then we
        # will not keep track of their identifiers (the essential point of doing function approximation to begin with).
        if update_state.i is not None:
            evaluated_states.add(update_state)

        # remove the time step from our n-step structure
        del t_state_a_g[update_t]

        # update the policy per num updates
        if num_updates_per_improvement is not None and q_S_A.update_count % num_updates_per_improvement == 0:
            q_S_A.improve_policy(
                agent=agent,
                states=evaluated_states,
                epsilon=epsilon
            )
