# Introduction
This is an implementation of concepts and algorithms described in "Reinforcement Learning: An Introduction" (Sutton
and Barto, 2018, 2nd edition). It is a work in progress, implemented with the following objectives in mind.

1. **Complete conceptual and algorithmic coverage**:  Implement all concepts and algorithms described in the text, plus 
some.
1. **Minimal dependencies**:  All computation specific to the text is implemented here.
1. **Complete test coverage**:  All implementations are paired with unit tests.
1. **General-purpose design**:  The text provides concise pseudocode that is not difficult to implement for the
examples covered; however, such implementations do not necessarily lead to reusable and extensible code that is 
generally applicable beyond such examples. The approach taken here should be generally applicable well beyond the text.

# Installation
This code is distributed via [PyPI](https://pypi.org/project/rlai/) and can be installed with `pip install rlai`.

# Case Studies
The gridworld and other simple environments (e.g., gambler's problem) are used throughout the package to develop, 
implement, and test algorithmic concepts. Sutton and Barto do a nice job of explaining how reinforcement learning works
for these environments. Below is a list of environments that are not covered in as much detail. They are more difficult
to train agents for and are instructive for understanding how agents are parameterized.
 
* OpenAI Gym
  * [Inverted Pendulum](https://matthewgerber.github.io/rlai/case_studies/inverted_pendulum.html)
  * [Acrobot](https://matthewgerber.github.io/rlai/case_studies/acrobot.html)
  * [Mountain Car](https://matthewgerber.github.io/rlai/case_studies/mountain_car.html)

# Figures
A list of figures can be found [here](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/figures). Most of these 
are reproductions of those shown in the text; however, even the reproductions typically provide detail not shown in the 
text.