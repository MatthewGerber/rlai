import logging

from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.gpi import PolicyImprovementEvent
from rlai.meta import rl_text
from rlai.policies.parameterized import ParameterizedPolicy
from rlai.utils import IncrementalSampleAverager, RunThreadManager
from rlai.value_estimation import StateActionValueEstimator


@rl_text(chapter=13, page=326)
def improve(
        agent: MdpAgent,
        policy: ParameterizedPolicy,
        environment: MdpEnvironment,
        num_episodes: int,
        update_upon_every_visit: bool,
        alpha: float,
        q_S_A: StateActionValueEstimator,
        thread_manager: RunThreadManager
):
    """
    Perform Monte Carlo improvement of an agent's policy within an environment via the REINFORCE policy gradient method.
    This improvement function operates over rewards obtained at the end of episodes, so it is only appropriate for
    episodic tasks.

    :param agent: Agent containing target policy to be optimized.
    :param policy: Parameterized policy to be optimized.
    :param environment: Environment.
    :param num_episodes: Number of episodes to execute.
    :param update_upon_every_visit: True to update each state-action pair upon each visit within an episode, or False to
    update each state-action pair upon the first visit within an episode.
    :param alpha: Policy gradient step size.
    :param thread_manager: Thread manager. The current function (and the thread running it) will wait on this manager
    before starting each iteration. This provides a mechanism for pausing, resuming, and aborting training. Omit for no
    waiting.
    :param q_S_A: Baseline state-action value estimator.
    """

    logging.info(f'Running Monte Carlo-based REINFORCE improvement for {num_episodes} episode(s).')

    baseline_a = None

    episode_reward_averager = IncrementalSampleAverager()
    episodes_per_print = max(1, int(num_episodes * 0.05))
    for episode_i in range(num_episodes):

        # reset the environment for the new run (always use the agent we're learning about, as state identifiers come
        # from it), and reset the agent accordingly.
        state = environment.reset_for_new_run(agent)
        agent.reset_for_new_run(state)

        # grab an arbitrary action if we're using a baseline.
        if q_S_A is not None:
            baseline_a = state.AA[0]

        # simulate until episode termination, keeping a trace of state-action pairs and their immediate rewards, as well
        # as the times of their first visits (only if we're doing first-visit evaluation).
        t = 0
        state_action_first_t = None if update_upon_every_visit else {}
        t_state_action_reward = []
        total_reward = 0.0
        while not state.terminal and (environment.T is None or t < environment.T):

            a = agent.act(t)
            state_a = (state, a)

            # mark time step of first visit, if we're doing first-visit evaluation.
            if state_action_first_t is not None and state_a not in state_action_first_t:
                state_action_first_t[state_a] = t

            next_state, next_reward = environment.advance(state, t, a, agent)
            t_state_action_reward.append((t, state_a, next_reward))
            total_reward += next_reward.r
            state = next_state
            t += 1

            agent.sense(state, t)

        # work backwards through the trace to calculate discounted returns. need to work backward in order for the value
        # of g at each time step t to be properly discounted.
        g = 0
        for t, state_a, reward in reversed(t_state_action_reward):

            g = agent.gamma * g + reward.r

            # if we're doing every-visit, or if the current time step was the first visit to the state-action, then g
            # is the discounted sample value. use it to update the policy.
            if state_action_first_t is None or state_action_first_t[state_a] == t:

                state, a = state_a

                # update the baseline state-value estimator. always pass the same arbitrary action to achieve state-
                # value estimation. as the text notes:  the baseline can be anything as long as it does not vary with
                # the action.
                if q_S_A is not None:
                    q_S_A[state][baseline_a].update(g)
                    q_S_A.improve_policy(agent, None, PolicyImprovementEvent.UPDATED_VALUE_ESTIMATE)
                    update_target = g - q_S_A[state][baseline_a].get_value()
                else:
                    update_target = g

                agent.pi.theta += agent.pi.get_update(a, state, alpha, update_target)

        episode_reward_averager.update(total_reward)

        episodes_finished = episode_i + 1
        if episodes_finished % episodes_per_print == 0:
            logging.info(f'Finished {episodes_finished} of {num_episodes} episode(s).')

    logging.info(f'Completed optimization. Average reward per episode:  {episode_reward_averager.get_value()}')

    for s in environment.SS:
        for a in s.AA:
            print(f'Pr({a}|{s}:  {agent.pi[s][a]}')
