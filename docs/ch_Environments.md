[Home](index.md) > Environments
### [rlai.core.environments.gymnasium.Gym](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/gymnasium.py#L93)
```
Generalized Gym environment. Any Gym environment can be executed by supplying the appropriate identifier.
```
### [rlai.core.environments.mancala.Mancala](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/mancala.py#L126)
```
Environment for the mancala game. This is a simple game with many rule variations, and it provides a greater
    challenge in terms of implementation and state-space size than the gridworld. I have implemented a fairly common
    variation summarized below.

    * One row of 6 pockets per player, each starting with 4 seeds.
    * Landing in the store earns another turn.
    * Landing in own empty pocket steals.
    * Game terminates when a player's pockets are clear.
    * Winner determined by store count.

    A couple of hours of Monte Carlo optimization explores more than 1 million states when playing against an
    equiprobable random opponent.
```
### [rlai.core.environments.mdp.ContinuousMdpEnvironment](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/mdp.py#L849)
```
MDP environment in which states and actions are continuous and multidimensional.
```
### [rlai.core.environments.network.TcpMdpEnvironment](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/network.py#L17)
```
An MDP environment served over a TCP connection from an external source (e.g., a simulation environment running as
    a separate program).
```
### [rlai.core.environments.robocode.RobocodeEnvironment](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/robocode.py#L176)
```
Robocode environment. The Java implementation of Robocode runs alongside the current environment, and a specialized
    robot implementation on the Java side makes TCP calls to the present Python class to exchange action and state
    information.
```
### [rlai.core.environments.robocode_continuous_action.RobocodeEnvironment](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/robocode_continuous_action.py#L185)
```
Robocode environment. The Java implementation of Robocode runs alongside the current environment, and a specialized
    robot implementation on the Java side makes TCP calls to the present Python class to exchange action and state
    information.
```
