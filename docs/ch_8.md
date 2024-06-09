[Home](index.md) > Chapter 8:  Planning and Learning with Tabular Methods
### [rlai.core.environments.mdp.MdpPlanningEnvironment](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/mdp.py#L424)
```
An MDP planning environment, used to generate simulated experience based on a model of the MDP that is learned
    through direct experience with the actual environment.
```
### [rlai.core.environments.mdp.EnvironmentModel](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/mdp.py#L262)
```
An environment model.
```
### [rlai.core.environments.mdp.PrioritizedSweepingMdpPlanningEnvironment](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/mdp.py#L504)
```
State-action transitions are prioritized based on the degree to which learning updates their values, and transitions
    with the highest priority are explored during planning.
```
### [rlai.core.environments.mdp.StochasticEnvironmentModel](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/mdp.py#L300)
```
A stochastic environment model.
```
### [rlai.core.environments.mdp.TrajectorySamplingMdpPlanningEnvironment](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/mdp.py#L734)
```
State-action transitions are selected by the agent based on the agent's policy, and the selected transitions are
    explored during planning.
```
