# Introduction
This is an implementation of concepts and algorithms described in "Reinforcement Learning: An Introduction" (Sutton
and Barto, 2018, 2nd edition). It is a work in progress, implemented with the following objectives in mind.

1. **Complete conceptual and algorithmic coverage**:  Implement all concepts and algorithms described in the text, plus some.
1. **Minimal dependencies**:  All computation specific to the text is implemented here.
1. **Complete test coverage**:  All implementations are paired with unit tests.
1. **General-purpose design**:  The text provides concise pseudocode that is not difficult to implement for the
examples covered; however, such implementations do not necessarily lead to reusable and extensible code that is 
generally applicable beyond such examples. The approach taken here should be generally applicable well beyond the text.

# Installation
This code is distributed via [PyPI](https://pypi.org/project/rlai/) and can be installed with `pip install rlai`.

# Figures
A list of figures can be found [here](../src/rlai/figures). Most of these are reproductions of those shown in the text; 
however, even the reproductions typically provide detail not shown in the text.

# General Topics
### [Environments](ch_Environments.md)
### [Training and Running Agents](ch_Training_and_Running_Agents.md)

# Book Chapters
### [Chapter 2:  Multi-armed Bandits](ch_2.md)
### [Chapter 3:  Finite Markov Decision Processes](ch_3.md)
### [Chapter 4:  Dynamic Programming](ch_4.md)
### [Chapter 5:  Monte Carlo Methods](ch_5.md)
### [Chapter 6:  Temporal-Difference Learning](ch_6.md)
### [Chapter 8:  Planning and Learning with Tabular Methods](ch_8.md)