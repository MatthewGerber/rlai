# Chapter 5:  Monte Carlo Methods
### [rlai.gpi.monte_carlo.evaluation.evaluate_v_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/monte_carlo/evaluation.py#L12)
```
Perform Monte Carlo evaluation of an agent's policy within an environment, returning state values. Uses a random
    action on the first time step to maintain exploration (exploring starts). This evaluation approach is only
    marginally useful in practice, as the state-value estimates require a model of the environmental dynamics (i.e.,
    the transition-reward probability distribution) in order to be applied. See `evaluate_q_pi` in this module for a
    more feature-rich and useful evaluation approach (i.e., state-action value estimation). This evaluation function
    operates over rewards obtained at the end of episodes, so it is only appropriate for episodic tasks.

    :param agent: Agent.
    :param environment: Environment.
    :param num_episodes: Number of episodes to execute.
    :return: Dictionary of MDP states and their estimated values under the agent's policy.
```
### [rlai.gpi.monte_carlo.evaluation.evaluate_q_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/monte_carlo/evaluation.py#L94)
```
Perform Monte Carlo evaluation of an agent's policy within an environment, returning state-action values. This
    evaluation function operates over rewards obtained at the end of episodes, so it is only appropriate for episodic
    tasks.

    :param agent: Agent containing target policy to be optimized.
    :param environment: Environment.
    :param num_episodes: Number of episodes to execute.
    :param exploring_starts: Whether or not to use exploring starts, forcing a random action in the first time step.
    This maintains exploration in the first state; however, unless each state has some nonzero probability of being
    selected as the first state, there is no assurance that all state-action pairs will be sampled. If the initial state
    is deterministic, consider passing False here and shifting the burden of exploration to the improvement step with
    a nonzero epsilon (see `rlai.gpi.improvement.improve_policy_with_q_pi`).
    :param update_upon_every_visit: True to update each state-action pair upon each visit within an episode, or False to
    update each state-action pair upon the first visit within an episode.
    :param off_policy_agent: Agent containing behavioral policy used to generate learning episodes. To ensure that the
    state-action value estimates converge to those of the target policy, the policy of the `off_policy_agent` must be
    soft (i.e., have positive probability for all state-action pairs that have positive probabilities in the agent's
    target policy).
    :param initial_q_S_A: Initial guess at state-action value, or None for no guess.
    :return: 3-tuple of (1) dictionary of all MDP states and their action-value averagers under the agent's policy, (2)
    set of only those states that were evaluated, and (3) the average reward obtained per episode.
```
### [rlai.gpi.monte_carlo.iteration.iterate_value_q_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/monte_carlo/iteration.py#L19)
```
Run Monte Carlo value iteration on an agent using state-action value estimates. This iteration function operates
    over rewards obtained at the end of episodes, so it is only appropriate for episodic tasks.

    :param agent: Agent.
    :param environment: Environment.
    :param num_improvements: Number of policy improvements to make.
    :param num_episodes_per_improvement: Number of policy evaluation episodes to execute for each iteration of
    improvement. Passing `1` will result in the Monte Carlo ES (Exploring Starts) algorithm.
    :param update_upon_every_visit: See `rlai.gpi.monte_carlo.evaluation.evaluate_q_pi`.
    :param epsilon: Total probability mass to spread across all actions, resulting in an epsilon-greedy policy. Must
    be >= 0 if provided.
    :param make_final_policy_greedy: Whether or not to make the agent's final policy greedy with respect to the q-values
    that have been learned, regardless of the value of epsilon used to estimate the q-values.
    :param off_policy_agent: See `rlai.gpi.monte_carlo.evaluation.evaluate_q_pi`. The policy of this agent will not
    updated by this function.
    :param num_improvements_per_plot: Number of improvements to make before plotting the per-improvement average. Pass
    None to turn off all plotting.
    :param num_improvements_per_checkpoint: Number of improvements per checkpoint save.
    :param checkpoint_path: Checkpoint path. Must be provided if `num_improvements_per_checkpoint` is provided.
    :param initial_q_S_A: Initial state-action value estimates (primarily useful for restarting from a checkpoint).
    :return: Dictionary of state-action value estimators.
```