# Chapter 6:  Temporal-Difference Learning
### [rlai.gpi.temporal_difference.evaluation.Mode](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/temporal_difference/evaluation.py#L15)
```
Modes of temporal-difference evaluation:  SARSA (on-policy), Q-Learning (off-policy), and Expected SARSA
    (off-policy).
```
### [rlai.gpi.temporal_difference.evaluation.evaluate_q_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/temporal_difference/evaluation.py#L33)
```
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
    :param planning_environment: Planning environment to learn through experience gained during evaluation, or None to
    not learn an environment model.
    :param initial_q_S_A: Initial guess at state-action value, or None for no guess.
    :return: 3-tuple of (1) dictionary of all MDP states and their action-value averagers under the agent's policy, (2)
    set of only those states that were evaluated, and (3) the average reward obtained per episode.
```
### [rlai.gpi.temporal_difference.iteration.iterate_value_q_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/temporal_difference/iteration.py#L18)
```
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
    :param planning_environment: Planning environment to learn and use.
    :param make_final_policy_greedy: Whether or not to make the agent's final policy greedy with respect to the q-values
    that have been learned, regardless of the value of epsilon used to estimate the q-values.
    :param num_improvements_per_plot: Number of improvements to make before plotting the per-improvement average. Pass
    None to turn off all plotting.
    :param num_improvements_per_checkpoint: Number of improvements per checkpoint save.
    :param checkpoint_path: Checkpoint path. Must be provided if `num_improvements_per_checkpoint` is provided.
    :param initial_q_S_A: Initial state-action value estimates (primarily useful for restarting from a checkpoint).
    :return: Dictionary of state-action value estimators.
```