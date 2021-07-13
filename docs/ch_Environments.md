# Environments
### [rlai.environments.mancala.Mancala](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/mancala.py#L127)
```
Environment for the mancala game. This is a simple game with many rule variations, and it provides a greater
    challenge in terms of implementation and state-space size than the gridworld. I have implemented a fairly common
    variation summarized below.

    * One row of 6 pockets per player, each starting with 4 seeds.
    * Landing in the store earns another turn.
    * Landing in own empty pocket steals.
    * Game terminates when a player's pockets are clear.
    * Winner determined by store count.

    A couple hours of Monte Carlo optimization explores more than 1 million states when playing against an equiprobable
    random opponent.
```
### [rlai.environments.network.TcpMdpEnvironment](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/network.py#L20)
```
An MDP environment served over a TCP connection from an external source (e.g., a simulation environment running as
    a separate program).
```
### [rlai.environments.openai_gym.Gym](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/openai_gym.py#L63)
```
Generalized Gym environment. Any OpenAI Gym environment can be executed by supplying the appropriate identifier.
```
### [rlai.environments.robocode.RobocodeEnvironment](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/robocode.py#L207)
```
Robocode environment. The Java implementation of Robocode runs alongside the current environment, and a specialized
    robot implementation on the Java side makes TCP calls to the present Python class to exchange action and state
    information.
```
