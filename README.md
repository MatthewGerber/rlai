# Table of Contents
- [Goals](#goals)
- [Chapter 2](#chapter-2)
  - [`rl.environments.bandit.Arm`](#rlenvironmentsbanditarm)
  - [`rl.agents.q_value.EpsilonGreedy`](#rlagentsq_valueepsilongreedy)
  - [`rl.agents.q_value.QValue`](#rlagentsq_valueqvalue)
  - [`rl.environments.bandit.KArmedBandit`](#rlenvironmentsbanditkarmedbandit)
  - [`rl.utils.IncrementalSampleAverager`](#rlutilsincrementalsampleaverager)
  - [`rl.agents.q_value.UpperConfidenceBound`](#rlagentsq_valueupperconfidencebound)
  - [`rl.agents.h_value.PreferenceGradient`](#rlagentsh_valuepreferencegradient)
- [Chapter 3](#chapter-3)
  - [`rl.environments.mdp.MdpEnvironment`](#rlenvironmentsmdpmdpenvironment)
  - [`rl.environments.mdp.Gridworld`](#rlenvironmentsmdpgridworld)
- [Chapter 4](#chapter-4)
  - [`rl.dynamic_programming.policy_evaluation.evaluate_v_pi`](#rldynamic_programmingpolicy_evaluationevaluate_v_pi)
  - [`rl.dynamic_programming.policy_evaluation.evaluate_q_pi`](#rldynamic_programmingpolicy_evaluationevaluate_q_pi)
  - [`rl.dynamic_programming.policy_improvement.improve_policy_with_q_pi`](#rldynamic_programmingpolicy_improvementimprove_policy_with_q_pi)
  - [`rl.dynamic_programming.policy_improvement.improve_policy_with_v_pi`](#rldynamic_programmingpolicy_improvementimprove_policy_with_v_pi)
  - [`rl.dynamic_programming.policy_iteration.iterate_policy_q_pi`](#rldynamic_programmingpolicy_iterationiterate_policy_q_pi)
  - [`rl.dynamic_programming.policy_iteration.iterate_policy_v_pi`](#rldynamic_programmingpolicy_iterationiterate_policy_v_pi)
  - [`rl.dynamic_programming.value_iteration.iterate_value_q_pi`](#rldynamic_programmingvalue_iterationiterate_value_q_pi)
  - [`rl.dynamic_programming.value_iteration.iterate_value_v_pi`](#rldynamic_programmingvalue_iterationiterate_value_v_pi)
<!--TOC-->

# Goals
The goals of this package are:

1. blah 
1. blah
1. blah

# Chapter 2
## `rl.environments.bandit.Arm`
```
Bandit arm.
```
## `rl.agents.q_value.EpsilonGreedy`
```
Nonassociative, epsilon-greedy agent.
```
## `rl.agents.q_value.QValue`
```
Nonassociative, q-value agent.
```
## `rl.environments.bandit.KArmedBandit`
```
K-armed bandit.
```
## `rl.utils.IncrementalSampleAverager`
```
An incremental, constant-time and -memory sample averager. Supports both decreasing (i.e., unweighted sample
    average) and constant (i.e., exponential recency-weighted average, pp. 32-33) step sizes.
```
## `rl.agents.q_value.UpperConfidenceBound`
```
Nonassociatve, upper-confidence-bound agent.
```
## `rl.agents.h_value.PreferenceGradient`
```
Preference-gradient agent.
```
# Chapter 3
## `rl.environments.mdp.MdpEnvironment`
```
MDP environment.
```
## `rl.environments.mdp.Gridworld`
```
Gridworld MDP environment.
```
# Chapter 4
## `rl.dynamic_programming.policy_evaluation.evaluate_v_pi`
```
Perform iterative policy evaluation of an agent's policy within an environment, returning state values.

    :param agent: MDP agent.
    :param environment: MDP environment.
    :param theta: Minimum tolerated change in state value estimates, below which evaluation terminates. Either `theta`
    or `num_iterations` (or both) can be specified, but passing neither will raise an exception.
    :param num_iterations: Number of evaluation iterations to execute.  Either `theta` or `num_iterations` (or both)
    can be specified, but passing neither will raise an exception.
    :param update_in_place: Whether or not to update value estimates in place.
    :param initial_v_S: Initial guess at state-value, or None for no guess.
    :return: Dictionary of MDP states and their estimated values.
```
## `rl.dynamic_programming.policy_evaluation.evaluate_q_pi`
```
Perform iterative policy evaluation of an agent's policy within an environment, returning state-action values.

    :param agent: MDP agent.
    :param environment: MDP environment.
    :param theta: Minimum tolerated change in state value estimates, below which evaluation terminates. Either `theta`
    or `num_iterations` (or both) can be specified, but passing neither will raise an exception.
    :param num_iterations: Number of evaluation iterations to execute.  Either `theta` or `num_iterations` (or both)
    can be specified, but passing neither will raise an exception.
    :param update_in_place: Whether or not to update value estimates in place.
    :param initial_q_S_A: Initial guess at state-action value, or None for no guess.
    :return: Dictionary of MDP states, actions, and their estimated values.
```
## `rl.dynamic_programming.policy_improvement.improve_policy_with_q_pi`
```
Improve an agent's policy according to its state-action value estimates. This makes the policy greedy with respect
    to the state-action value estimates. In cases where multiple such greedy actions exist for a state, each of the
    greedy actions will be assigned equal probability.

    :param agent: Agent.
    :param q_pi: State-action value estimates for the agent's policy.
    :return: True if policy was changed and False if the policy was not changed.
```
## `rl.dynamic_programming.policy_improvement.improve_policy_with_v_pi`
```
Improve an agent's policy according to its state-value estimates. This makes the policy greedy with respect to the
    state-value estimates. In cases where multiple such greedy actions exist for a state, each of the greedy actions
    will be assigned equal probability.

    :param agent: Agent.
    :param environment: Environment.
    :param v_pi: State-value estimates for the agent's policy.
    :return: True if policy was changed and False if the policy was not changed.
```
## `rl.dynamic_programming.policy_iteration.iterate_policy_q_pi`
```
Run policy iteration on an agent using state-value estimates.

    :param agent: Agent.
    :param environment: Environment.
    :param theta: See `evaluate_q_pi`.
    :param update_in_place: See `evaluate_q_pi`.
    :return: Final state-action value estimates.
```
## `rl.dynamic_programming.policy_iteration.iterate_policy_v_pi`
```
Run policy iteration on an agent using state-value estimates.

    :param agent: Agent.
    :param environment: Environment.
    :param theta: See `evaluate_v_pi`.
    :param update_in_place: See `evaluate_v_pi`.
    :return: Final state-value estimates.
```
## `rl.dynamic_programming.value_iteration.iterate_value_q_pi`
```
Run policy iteration on an agent using state-value estimates.

    :param agent: Agent.
    :param environment: Environment.
    :param evaluation_iterations_per_improvement: Number of policy evaluation iterations to execute for each iteration
    of improvement (e.g., passing 1 results in Equation 4.10).
    :param update_in_place: See `evaluate_v_pi`.
    :return: Final state-action value estimates.
```
## `rl.dynamic_programming.value_iteration.iterate_value_v_pi`
```
Run value iteration on an agent using state-value estimates.

    :param agent: Agent.
    :param environment: Environment.
    :param evaluation_iterations_per_improvement: Number of policy evaluation iterations to execute for each iteration
    of improvement (e.g., passing 1 results in Equation 4.10).
    :param update_in_place: See `evaluate_v_pi`.
    :return: Final state-value estimates.
```
