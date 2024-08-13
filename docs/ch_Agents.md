[Home](index.md) > Agents
### [rlai.core.Agent](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/__init__.py#L404)
```
Base class for all agents.
```
### [rlai.core.Human](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/__init__.py#L1302)
```
An interactive, human-driven agent that prompts for actions at each time step.
```
### [rlai.core.MdpAgent](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/__init__.py#L1151)
```
MDP agent. Adds the concepts of state, reward discounting, and policy-based action to the base agent.
```
### [rlai.core.StochasticMdpAgent](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/__init__.py#L1233)
```
Stochastic MDP agent. Adds random selection of actions based on probabilities specified in the agent's policy.
```
### [rlai.core.environments.robocode.RobocodeAgent](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/robocode.py#L59)
```
Robocode agent.
```
### [rlai.core.environments.robocode_continuous_action.RobocodeAgent](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/robocode_continuous_action.py#L63)
```
Robocode agent.
```
### [rlai.gpi.state_action_value.ActionValueMdpAgent](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/state_action_value/__init__.py#L326)
```
A stochastic MDP agent whose policy is based on action-value estimation. This agent is generally appropriate for
    discrete and continuous state spaces in which we estimate the value of actions using tabular and
    function-approximation methods, respectively. The action space need to be discrete in all of these cases. If the
    action space is continuous, then consider the `ParameterizedMdpAgent`.
```
### [rlai.policy_gradient.ParameterizedMdpAgent](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/policy_gradient/__init__.py#L14)
```
A stochastic MDP agent whose policy is directly parameterized. This agent is generally appropriate when both the
    state and action spaces are continuous. If the action space is discrete, then consider the `ActionValueMdpAgent`.
```
