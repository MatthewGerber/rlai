# Chapter 8:  Planning and Learning with Tabular Methods
### [rlai.environments.mdp.MdpPlanningEnvironment](../src/rlai/environments/mdp.py)
```
An MDP planning environment, used to generate simulated experience based on a model of the MDP that is learned
    through direct experience with the actual environment.
```
### [rlai.planning.environment_models.EnvironmentModel](../src/rlai/planning/environment_models.py)
```
An environment model.
```
### [rlai.environments.mdp.PrioritizedSweepingMdpPlanningEnvironment](../src/rlai/environments/mdp.py)
```
State-action transitions are prioritized based on the degree to which learning updates their values, and transitions
    with the highest priority are explored during planning.
```
### [rlai.planning.environment_models.StochasticEnvironmentModel](../src/rlai/planning/environment_models.py)
```
A stochastic environment model.
```
### [rlai.environments.mdp.TrajectorySamplingMdpPlanningEnvironment](../src/rlai/environments/mdp.py)
```
State-action transitions are selected by the agent based on the agent's policy, and the selected transitions are
    explored during planning.
```
