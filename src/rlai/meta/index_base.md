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

# Installation, Use, and Development
The `rlai` code is distributed via [PyPI](https://pypi.org/project/rlai/) and can be installed with `pip install rlai`. 
There are several ways to use the package.

* Incorporate `rlai` into another project:  See the 
  [example repository](https://github.com/MatthewGerber/rlai-dependency-example) for how a project can be structured to 
  consume the `rlai` package functionality within source code.
  
* Use `rlai` directly
    * Most of the `rlai` functionality is exposed via the companion JupyterLab notebook. See the 
      [JupyterLab guide](jupyterlab_guide.md) for more information. 
    * Using `rlai` from the command-line interface (CLI) is demonstrated in the case studies below and is also explored 
      in the [CLI guide](cli_guide.md).

Looking for a place to dig in? Below are a few ideas organized by area of interest.

* Explore new OpenAI Gym environments:  OpenAI Gym provides a wide range of interesting environments, and
  experimenting with them can be as simple as modifying an existing training command (e.g., the one for
  [inverted pendulum](https://matthewgerber.github.io/rlai/case_studies/inverted_pendulum.html)), replacing the 
  `--gym-id` with something else. Other changes might be needed depending on the environment, but Gym is particularly
  convenient.

* Incorporate new statistical learning methods:  The `rlai` 
  [SKLearnSGD](https://github.com/MatthewGerber/rlai/blob/master/src/rlai/value_estimation/function_approximation/models/sklearn.py)
  module demonstrates how to use methods in scikit-learn (in this case stochastic gradient descent regression) to 
  approximate state-action value functions. This is just one approach, and it would be interesting to compare time, 
  memory, and reward performance with a nonparametric approach like 
  [KNN regression](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html).
  
Feel free to submit issues and pull requests.
  
# Features
* Diagnostic and interpretation tools:  Diagnostic and interpretation tools become critical as the environment and agent 
  increase in complexity (e.g., from tabular methods in small, discrete-space gridworlds to value function approximation 
  methods in large, continuous-space control problems). Such tools can be found 
  [here](model_diagnostics_and_interpretation.md).

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
are reproductions of those shown in the Sutton and Barto text; however, even the reproductions typically provide detail 
not shown in the text.
