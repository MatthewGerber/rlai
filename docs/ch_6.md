[Home](index.md) > Chapter 6:  Temporal-Difference Learning
### [rlai.gpi.temporal_difference.evaluation.Mode](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/temporal_difference/evaluation.py#L17)
```
Modes of temporal-difference evaluation:  SARSA (on-policy), Q-Learning (off-policy), and Expected SARSA
    (off-policy).
```
### [rlai.gpi.temporal_difference.evaluation.evaluate_q_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/temporal_difference/evaluation.py#L36)
```
Perform temporal-difference (TD) evaluation of an agent's policy within an environment, returning state-action
    values. This evaluation function implements both on-policy TD learning (SARSA) and off-policy TD learning
    (Q-learning and expected SARSA), and n-step updates are implemented for all learning modes.

    :param agent: Agent containing target policy to be optimized.
    :param environment: Environment.
    :param num_episodes: Number of episodes to execute.
    :param num_updates_per_improvement: Number of state-action value updates to execute for each iteration of policy
    improvement, or None for policy improvement per specified number of episodes.
    :param alpha: Constant step size to use when updating Q-values, or None for 1/n step size.
    :param mode: Evaluation mode (see `rlai.gpi.temporal_difference.evaluation.Mode`).
    :param n_steps: Number of steps to accumulate rewards before updating estimated state-action values. Must be in the
    range [1, inf], or None for infinite step size (Monte Carlo evaluation).
    :param planning_environment: Planning environment to learn through experience gained during evaluation, or None to
    not learn an environment model.
    :return: 2-tuple of (1) set of only those states that were evaluated, and (2) the average reward obtained per
    episode.
```
### [rlai.gpi.temporal_difference.iteration.iterate_value_q_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/temporal_difference/iteration.py#L19)
```
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
    :param planning_environment: Planning environment to learn and use.
    :param make_final_policy_greedy: Whether to make the agent's final policy greedy with respect to the q-values
    that have been learned, regardless of the value of epsilon used to estimate the q-values.
    :param thread_manager: Thread manager. The current function (and the thread running it) will wait on this manager
    before starting each iteration. This provides a mechanism for pausing, resuming, and aborting training. Omit for no
    waiting.
    :param num_improvements_per_plot: Number of improvements to make before plotting the per-improvement average. Pass
    None to turn off all plotting.
    :param num_improvements_per_checkpoint: Number of improvements per checkpoint save.
    :param checkpoint_path: Checkpoint path. Must be provided if `num_improvements_per_checkpoint` is provided.
    :param pdf_save_path: Path where a PDF of all plots is to be saved, or None for no PDF.
    :return: Final checkpoint path, or None if checkpoints were not saved.
```
