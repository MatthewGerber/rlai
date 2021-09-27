# RLAI
* Content
{:toc}
  
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
   
# Quick Start
For single-click access to a graphical interface for RLAI, please click below:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MatthewGerber/rlai/HEAD?urlpath=lab/tree/jupyter/trainer.ipynb)

Note that Binder notebooks are hosted for free by sponsors who donate computational infrastructure. Limitations are 
placed on each notebook, so don't expect the Binder interface to support heavy workloads. See the following section for
alternatives.

# Installation, Use, and Development
The RLAI code is distributed via [PyPI](https://pypi.org/project/rlai/) and can be installed with `pip install rlai`. 
There are several ways to use the package.

* JupyterLab notebook:  Most of the RLAI functionality is exposed via the companion JupyterLab notebook. See the 
  [JupyterLab guide](jupyterlab_guide.md) for more information.  

* Package dependency:  See the [example repository](https://github.com/MatthewGerber/rlai-dependency-example) for how a 
  project can be structured to consume the RLAI package functionality within source code.
  
* Command-line interface:  Using RLAI from the command-line interface (CLI) is demonstrated in the case studies below 
  and is also explored in the [CLI guide](cli_guide.md).

Looking for a place to dig in? Below are a few ideas organized by area of interest.

* Explore new OpenAI Gym environments:  OpenAI Gym provides a wide range of interesting environments, and
  experimenting with them can be as simple as modifying an existing training command (e.g., the one for
  [inverted pendulum](case_studies/inverted_pendulum.md)) and replacing the 
  `--gym-id` with something else. Other changes might be needed depending on the environment, but Gym is particularly
  convenient.

* Incorporate new statistical learning methods:  The RLAI 
  [SKLearnSGD](https://github.com/MatthewGerber/rlai/blob/master/src/rlai/value_estimation/function_approximation/models/sklearn.py)
  module demonstrates how to use methods in scikit-learn (in this case stochastic gradient descent regression) to 
  approximate state-action value functions. This is just one approach, and it would be interesting to compare time, 
  memory, and reward performance with a nonparametric approach like 
  [KNN regression](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html).
  
* Feel free to [ask questions](https://github.com/MatthewGerber/rlai/discussions), 
  [submit issues](https://github.com/MatthewGerber/rlai/issues), and 
  [submit pull requests](https://github.com/MatthewGerber/rlai/pulls).
  
# Features
* Diagnostic and interpretation tools:  Diagnostic and interpretation tools become critical as the environment and agent 
  increase in complexity (e.g., from tabular methods in small, discrete-space gridworlds to value function approximation 
  methods in large, continuous-space control problems). Such tools can be found 
  [here](model_diagnostics_and_interpretation.md).

# Case Studies
The gridworld and other simple environments (e.g., gambler's problem) are used throughout the package to develop, 
implement, and test algorithmic concepts. Sutton and Barto do a nice job of explaining how reinforcement learning works
for these environments. Below is a list of environments that are not covered in as much detail (e.g., the mountain car)
or are not covered at all (e.g., Robocode). They are more difficult to train agents for and are instructive for 
understanding how agents are parameterized and rewarded.

## OpenAI Gym
[OpenAI Gym](https://gym.openai.com) is a collection of environments that range from traditional control to advanced 
robotics. Case studies have been developed for the following OpenAI Gym environments, which are ordered roughly by 
increasing complexity:

* [Inverted Pendulum](case_studies/inverted_pendulum.md)
* [Acrobot](case_studies/acrobot.md)
* [Mountain Car](case_studies/mountain_car.md)
* [Mountain Car with Continuous Control](case_studies/mountain_car_continuous.md)
* [Lunar Lander with Continuous Control](case_studies/lunar_lander_continuous.md)

## Robocode
[Robocode](https://github.com/robo-code/robocode) is a simulation-based robotic combat programming game with a 
dynamically rich environment, multi-agent teaming, and a large user community. Read more 
[here](case_studies/robocode.md).

# Figures from the Textbook
A list of figures can be found [here](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/figures). Most of these 
are reproductions of those shown in the Sutton and Barto text; however, even the reproductions typically provide detail 
not shown in the text.

# Links to Code by Topic
### [Actions](ch_Actions.md)
### [Agents](ch_Agents.md)
### [Diagnostics](ch_Diagnostics.md)
### [Environments](ch_Environments.md)
### [Feature Extractors](ch_Feature_Extractors.md)
### [Rewards](ch_Rewards.md)
### [States](ch_States.md)
### [Training and Running Agents](ch_Training_and_Running_Agents.md)
### [Value Estimation](ch_Value_Estimation.md)

# Links to Code by Book Chapter
### [Chapter 1:  Introduction](ch_1.md)
### [Chapter 2:  Multi-armed Bandits](ch_2.md)
### [Chapter 3:  Finite Markov Decision Processes](ch_3.md)
### [Chapter 4:  Dynamic Programming](ch_4.md)
### [Chapter 5:  Monte Carlo Methods](ch_5.md)
### [Chapter 6:  Temporal-Difference Learning](ch_6.md)
### [Chapter 8:  Planning and Learning with Tabular Methods](ch_8.md)
### [Chapter 9:  On-policy Prediction with Approximation](ch_9.md)
### [Chapter 10:  On-policy Control with Approximation](ch_10.md)
### [Chapter 13:  Policy Gradient Methods](ch_13.md)
