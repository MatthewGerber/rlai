# Introduction
This is an implementation of concepts and algorithms described in "Reinforcement Learning: An Introduction" (Sutton
and Barto, 2018, 2nd edition). It is a work in progress that I started as a personal hobby, and its state reflects the 
extent to which I have progressed through the text. I have implemented it with the following objectives in mind.

1. **Complete conceptual and algorithmic coverage**:  Implement all concepts and algorithms described in the text, plus some.
1. **Minimal dependencies**:  All computation specific to the text is implemented here.
1. **Complete test coverage**:  All implementations are paired with unit tests.
1. **Clean object-oriented design**:  The text often provides concise pseudocode that is not difficult to write a one-off 
program for; however, it is an altogether different matter to architect a reusable and extensible codebase that achieves
the goals listed above in an object-oriented fashion.

As with all objectives, none of the above are fully realized. In particular, (1) is not met since I decided to make this 
repository public well before finishing. But the remaining objectives are fairly well satisfied.

# Figures
A list of figures can be found [here](src/rl/figures). Most of these are reproductions of those shown in the text; 
however, even the reproductions typically provide detail not shown in the text.

# Content
The following sections are generated programatically from annotation markers placed on functions and classes within the 
code. They give a rough sense of what is currently implemented, with respect to the text. Topics beyond those in the 
text are summarized below.

## Mancala
This is a simple game with many rule variations, and it provides a greater challenge in terms of implementation and 
state-space size than the gridworld. I have implemented a fairly common variation summarized below.

* One row of 6 pockets per player, each starting with 4 seeds.
* Landing in the store earns another turn.
* Landing in own empty pocket steals.
* Game terminates when a player's pockets are clear.
* Winner determined by store count.

A couple hours of Monte Carlo optimization explores more than 1 million states when playing against an equiprobable 
random opponent.

![mancala](src/rl/figures/Mancala Learning.png)

Key files are listed below.

* [Environment](src/rl/environments/mancala.py)
* [Test](test/rl/environments/mancala_test.py)