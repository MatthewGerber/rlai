[Home](index.md) > Chapter 4:  Dynamic Programming
### [rlai.gpi.dynamic_programming.evaluation.evaluate_v_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/dynamic_programming/evaluation.py#L14)
```
Perform iterative policy evaluation of an agent's policy within an environment, returning state values.

    :param agent: MDP agent. Must contain a policy `pi` that has been fully initialized with instances of
    `rlai.core.ModelBasedMdpState`.
    :param environment: Model-based MDP environment to evaluate.
    :param theta: Minimum tolerated change in state-value estimates, below which evaluation terminates. Either `theta`
    or `num_iterations` (or both) can be specified, but passing neither will raise an exception.
    :param num_iterations: Number of evaluation iterations to execute.  Either `theta` or `num_iterations` (or both)
    can be specified, but passing neither will raise an exception.
    :param update_in_place: Whether to update value estimates in place.
    :param initial_v_S: Initial guess at state-value, or None for no guess.
    :return: 2-tuple of (1) dictionary of MDP states and their estimated values under the agent's policy, and (2) final
    value of delta.
```
### [rlai.gpi.dynamic_programming.evaluation.evaluate_q_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/dynamic_programming/evaluation.py#L109)
```
Perform iterative policy evaluation of an agent's policy within an environment, returning state-action values.

    :param agent: MDP agent.
    :param environment: Model-based MDP environment to evaluate.
    :param theta: Minimum tolerated change in state-value estimates, below which evaluation terminates. Either `theta`
    or `num_iterations` (or both) can be specified, but passing neither will raise an exception.
    :param num_iterations: Number of evaluation iterations to execute.  Either `theta` or `num_iterations` (or both)
    can be specified, but passing neither will raise an exception.
    :param update_in_place: Whether to update value estimates in place.
    :param initial_q_S_A: Initial guess at state-action value, or None for no guess.
    :return: 2-tuple of (1) dictionary of MDP states, actions, and their estimated values under the agent's policy, and
    (2) final value of delta.
```
### [rlai.gpi.dynamic_programming.iteration.iterate_policy_q_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/dynamic_programming/iteration.py#L65)
```
Run policy iteration on an agent using state-value estimates.

    :param agent: MDP agent. Must contain a policy `pi` that has been fully initialized with instances of
    `rlai.core.ModelBasedMdpState`.
    :param environment: Model-based MDP environment to evaluate.
    :param theta: Minimum tolerated change in state-value estimates, below which evaluation terminates.
    :param update_in_place: Whether to update value estimates in place.
    :return: Final state-action value estimates.
```
### [rlai.gpi.dynamic_programming.iteration.iterate_policy_v_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/dynamic_programming/iteration.py#L12)
```
Run policy iteration on an agent using state-value estimates.

    :param agent: MDP agent. Must contain a policy `pi` that has been fully initialized with instances of
    `rlai.core.ModelBasedMdpState`.
    :param environment: Model-based MDP environment to evaluate.
    :param theta: Minimum tolerated change in state-value estimates, below which evaluation terminates.
    :param update_in_place: Whether to update value estimates in place.
    :return: Final state-value estimates.
```
### [rlai.gpi.dynamic_programming.iteration.iterate_value_v_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/dynamic_programming/iteration.py#L114)
```
Run dynamic programming value iteration on an agent using state-value estimates.

    :param agent: MDP agent. Must contain a policy `pi` that has been fully initialized with instances of
    `rlai.core.ModelBasedMdpState`.
    :param environment: Model-based MDP environment to evaluate.
    :param theta: See `evaluate_v_pi`.
    :param evaluation_iterations_per_improvement: Number of policy evaluation iterations to execute for each iteration
    of improvement (e.g., passing 1 results in Equation 4.10).
    :param update_in_place: See `evaluate_v_pi`.
    :return: Final state-value estimates.
```
### [rlai.core.environments.gamblers_problem.GamblersProblem](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/gamblers_problem.py#L13)
```
Gambler's problem MDP environment.
```
### [rlai.gpi.dynamic_programming.iteration.iterate_value_q_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/dynamic_programming/iteration.py#L172)
```
Run value iteration on an agent using state-action value estimates.

    :param agent: MDP agent. Must contain a policy `pi` that has been fully initialized with instances of
    `rlai.core.ModelBasedMdpState`.
    :param environment: Model-based MDP environment to evaluate.
    :param theta: See `evaluate_q_pi`.
    :param evaluation_iterations_per_improvement: Number of policy evaluation iterations to execute for each iteration
    of improvement.
    :param update_in_place: See `evaluate_q_pi`.
    :return: Final state-action value estimates.
```
