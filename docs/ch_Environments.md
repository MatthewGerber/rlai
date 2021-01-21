# Environments
### [rlai.environments.mancala.Mancala](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/mancala.py#L125)
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
### [rlai.environments.openai_gym.Gym](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/openai_gym.py#L62)
```
Generalized Gym environment. Any OpenAI Gym environment can be executed by supplying the appropriate identifier.
```
