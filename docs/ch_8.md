[Home](index.md) > Chapter 8:  Planning and Learning with Tabular Methods
### [rlai.core.environments.mdp.MdpPlanningEnvironment](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/mdp.py#L257)
```
An MDP planning environment, used to generate simulated experience based on a model of the MDP that is learned
    through direct experience with the actual environment.
```
### [rlai.planning.environment_models.EnvironmentModel](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/planning/environment_models.py#L15)
```
An environment model.
```
### [rlai.core.environments.mdp.PrioritizedSweepingMdpPlanningEnvironment](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/mdp.py#L337)
```
State-action transitions are prioritized based on the degree to which learning updates their values, and transitions
    with the highest priority are explored during planning.
```
### [rlai.planning.environment_models.StochasticEnvironmentModel](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/planning/environment_models.py#L53)
```
A stochastic environment model.
```
### [rlai.core.environments.mdp.TrajectorySamplingMdpPlanningEnvironment](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/mdp.py#L560)
```
State-action transitions are selected by the agent based on the agent's policy, and the selected transitions are
    explored during planning.
```
