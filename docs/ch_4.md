# Chapter 4:  Dynamic Programming
### [rlai.gpi.dynamic_programming.evaluation.evaluate_v_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/dynamic_programming/evaluation.py#L16)
```
Perform iterative policy evaluation of an agent's policy within an environment, returning state values.

    :param agent: MDP agent. Must contain a policy `pi` that has been fully initialized with instances of
    `rlai.states.mdp.ModelBasedMdpState`.
    :param environment: Model-based MDP environment to evaluate.
    :param theta: Minimum tolerated change in state-value estimates, below which evaluation terminates. Either `theta`
    or `num_iterations` (or both) can be specified, but passing neither will raise an exception.
    :param num_iterations: Number of evaluation iterations to execute.  Either `theta` or `num_iterations` (or both)
    can be specified, but passing neither will raise an exception.
    :param update_in_place: Whether or not to update value estimates in place.
    :param initial_v_S: Initial guess at state-value, or None for no guess.
    :return: 2-tuple of (1) dictionary of MDP states and their estimated values under the agent's policy, and (2) final
    value of delta.
```
### [rlai.gpi.dynamic_programming.evaluation.evaluate_q_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/dynamic_programming/evaluation.py#L103)
```
Perform iterative policy evaluation of an agent's policy within an environment, returning state-action values.

    :param agent: MDP agent.
    :param environment: Model-based MDP environment to evaluate.
    :param theta: Minimum tolerated change in state-value estimates, below which evaluation terminates. Either `theta`
    or `num_iterations` (or both) can be specified, but passing neither will raise an exception.
    :param num_iterations: Number of evaluation iterations to execute.  Either `theta` or `num_iterations` (or both)
    can be specified, but passing neither will raise an exception.
    :param update_in_place: Whether or not to update value estimates in place.
    :param initial_q_S_A: Initial guess at state-action value, or None for no guess.
    :return: 2-tuple of (1) dictionary of MDP states, actions, and their estimated values under the agent's policy, and
    (2) final value of delta.
```
### [rlai.gpi.dynamic_programming.improvement.improve_policy_with_v_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/dynamic_programming/improvement.py#L12)
```
Improve an agent's policy according to its state-value estimates. This makes the policy greedy with respect to the
    state-value estimates. In cases where multiple such greedy actions exist for a state, each of the greedy actions
    will be assigned equal probability.

    Note that the present function resides within `rlai.gpi.dynamic_programming.improvement` and requires state-value
    estimates of states that are model-based. These are the case because policy improvement from state values is only
    possible if we have a model of the environment. Compare with `rlai.gpi.improvement.improve_policy_with_q_pi`, which
    accepts model-free states since state-action values are estimated directly.

    :param agent: Agent.
    :param environment: Model-based environment.
    :param v_pi: State-value estimates for the agent's policy.
    :return: Number of states in which the policy was improved.
```
### [rlai.gpi.improvement.improve_policy_with_q_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/improvement.py#L13)
```
Improve an agent's policy according to its state-action value estimates. This makes the policy greedy with respect
    to the state-action value estimates. In cases where multiple such greedy actions exist for a state, each of the
    greedy actions will be assigned equal probability.

    :param agent: Agent.
    :param q_pi: State-action value estimates for the agent's policy.
    :param epsilon: Total probability mass to divide across all actions for a state, resulting in an epsilon-greedy
    policy. Must be >= 0.0 if given. Pass None to generate a purely greedy policy.
    :return: Number of states in which the policy was improved.
```
### [rlai.gpi.dynamic_programming.iteration.iterate_policy_q_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/dynamic_programming/iteration.py#L62)
```
Run policy iteration on an agent using state-value estimates.

    :param agent: MDP agent. Must contain a policy `pi` that has been fully initialized with instances of
    `rlai.states.mdp.ModelBasedMdpState`.
    :param environment: Model-based MDP environment to evaluate.
    :param theta: See `evaluate_q_pi`.
    :param update_in_place: See `evaluate_q_pi`.
    :return: Final state-action value estimates.
```
### [rlai.gpi.dynamic_programming.iteration.iterate_policy_v_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/dynamic_programming/iteration.py#L15)
```
Run policy iteration on an agent using state-value estimates.

    :param agent: MDP agent. Must contain a policy `pi` that has been fully initialized with instances of
    `rlai.states.mdp.ModelBasedMdpState`.
    :param environment: Model-based MDP environment to evaluate.
    :param theta: See `evaluate_v_pi`.
    :param update_in_place: See `evaluate_v_pi`.
    :return: Final state-value estimates.
```
### [rlai.gpi.dynamic_programming.iteration.iterate_value_v_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/dynamic_programming/iteration.py#L108)
```
Run dynamic programming value iteration on an agent using state-value estimates.

    :param agent: MDP agent. Must contain a policy `pi` that has been fully initialized with instances of
    `rlai.states.mdp.ModelBasedMdpState`.
    :param environment: Model-based MDP environment to evaluate.
    :param theta: See `evaluate_v_pi`.
    :param evaluation_iterations_per_improvement: Number of policy evaluation iterations to execute for each iteration
    of improvement (e.g., passing 1 results in Equation 4.10).
    :param update_in_place: See `evaluate_v_pi`.
    :return: Final state-value estimates.
```
### [rlai.environments.gamblers_problem.GamblersProblem](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/gamblers_problem.py#L16)
```
Gambler's problem MDP environment.
```
### [rlai.gpi.dynamic_programming.iteration.iterate_value_q_pi](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/dynamic_programming/iteration.py#L160)
```
Run value iteration on an agent using state-action value estimates.

    :param agent: MDP agent. Must contain a policy `pi` that has been fully initialized with instances of
    `rlai.states.mdp.ModelBasedMdpState`.
    :param environment: Model-based MDP environment to evaluate.
    :param theta: See `evaluate_q_pi`.
    :param evaluation_iterations_per_improvement: Number of policy evaluation iterations to execute for each iteration
    of improvement.
    :param update_in_place: See `evaluate_q_pi`.
    :return: Final state-action value estimates.
```
