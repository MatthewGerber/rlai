# RLAI
* Content
{:toc}

# Introduction
This is an implementation of concepts and algorithms described in "Reinforcement Learning: An Introduction" (Sutton
and Barto, 2018, 2nd edition). It is a work in progress, implemented with the following objectives in mind.

1. **Complete conceptual and algorithmic coverage**:  Implement all concepts and algorithms described in the text, plus 
some.
2. **Minimal dependencies**:  All computation specific to the text is implemented here.
3. **Complete test coverage**:  All implementations are paired with unit tests.
4. **General-purpose design**:  The text provides concise pseudocode that is not difficult to implement for the
examples covered; however, such implementations do not necessarily lead to reusable and extensible code that is 
generally applicable beyond such examples. The approach taken here should be generally applicable well beyond the text.

# Status
* [![PyPI version](https://badge.fury.io/py/rlai.svg)](https://badge.fury.io/py/rlai)
* [![Run Python Tests](https://github.com/MatthewGerber/rlai/actions/workflows/run-tests-on-push.yml/badge.svg)](https://github.com/MatthewGerber/rlai/actions/workflows/run-tests-on-push.yml)
* [![Coverage Status](https://coveralls.io/repos/github/MatthewGerber/rlai/badge.svg?branch=master)](https://coveralls.io/github/MatthewGerber/rlai?branch=master)

# Quick Start
For single-click access to a graphical interface for RLAI, please click below:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MatthewGerber/rlai/HEAD?urlpath=lab/tree/jupyter/trainer.ipynb)

Note that Binder notebooks are hosted for free by sponsors who donate computational infrastructure. Limitations are 
placed on each notebook, so don't expect the Binder interface to support heavy workloads. See the following section for
alternatives.

# Installation and Use
RLAI requires `swig` and `ffmpeg` to be installed on the system. These can be installed using a package manager on your
OS (e.g., Homebrew for macOS, `apt` for Ubuntu, etc.). If installing with Homebrew on macOS, then you might need to add 
an environment variable pointing to ffmpeg as follows:
```shell
echo 'export IMAGEIO_FFMPEG_EXE="/opt/homebrew/bin/ffmpeg"' >> ~/.bash_profile
```

The RLAI code is distributed via [PyPI](https://pypi.org/project/rlai/). There are several ways to use the package.

* JupyterLab notebook:  Most of the RLAI functionality is exposed via the companion JupyterLab notebook. See the 
  [JupyterLab guide](jupyterlab_guide.md) for more information.  

* Package dependency:  See the [example repository](https://github.com/MatthewGerber/rlai-dependency-example) for how a 
  project can be structured to consume the RLAI package functionality within source code.
  
* Command-line interface:  Using RLAI from the command-line interface (CLI) is demonstrated in the case studies below 
  and is also explored in the [CLI guide](cli_guide.md).

* See [here](raspberry_pi.md) for how to use RLAI on a Raspberry Pi system. 

# Development
Looking for a place to dig in? Below are a few ideas organized by area of interest.

* Explore new Gym environments:  Gym provides a wide range of interesting environments, and
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

## Gymnasium
[Gymnasium](https://gymnasium.farama.org) is a collection of environments that range from traditional control to 
advanced robotics. Case studies have been developed for the following environments, which are ordered roughly by 
increasing complexity:

* [Inverted Pendulum](case_studies/inverted_pendulum.md)
* [Acrobot](case_studies/acrobot.md)
* [Mountain Car](case_studies/mountain_car.md)
* [Mountain Car with Continuous Control](case_studies/mountain_car_continuous.md)
* [Lunar Lander with Continuous Control](case_studies/lunar_lander_continuous.md)
* [MuJoCo Swimming Worm with Continuous Control](case_studies/mujoco_swimming_worm.md) 
  * A follow-up using [process-level parallelization](case_studies/mujoco_swimming_worm_pooled.md) for faster, better 
    results.
  * See the MuJoCo section below for tips on installing MuJoCo.

## MuJoCo
RLAI works with [MuJoCo](https://mujoco.org/) either via Gymnasium described above or directly via the 
MuJoCo-provided Python bindings. On macOS, see [here](https://stackoverflow.com/questions/63475461/unable-to-import-opengl-gl-in-python-on-macos)
for how to fix OpenGL errors.

## Robocode
[Robocode](https://github.com/robo-code/robocode) is a simulation-based robotic combat programming game with a 
dynamically rich environment, multi-agent teaming, and a large user community. Read more 
[here](case_studies/robocode.md).
   
# Figures from the Textbook
A list of figures can be found [here](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/figures). Most of these 
are reproductions of those shown in the Sutton and Barto text; however, even the reproductions typically provide detail 
not shown in the text.

# Links to Code
See [here](links_to_code.md).

# Incrementing and Tagging Versions with Poetry
1. Begin the next prerelease number within the current prerelease phase (e.g., `0.1.0a0` → `0.1.0a1`):
   ```shell
   OLD_VERSION=$(poetry version --short)
   poetry version prerelease
   VERSION=$(poetry version --short)
   git commit -a -m "Next prerelease number:  ${OLD_VERSION} → ${VERSION}"
   git push
   ```
2. Begin the next prerelease phase (e.g., `0.1.0a1` → `0.1.0b0`):
   ```shell
   OLD_VERSION=$(poetry version --short)
   poetry version prerelease --next-phase
   VERSION=$(poetry version --short)
   git commit -a -m "Next prerelease phase:  ${OLD_VERSION} → ${VERSION}"
   git push
   ```
   The phases progress as alpha (`a`), beta (`b`), and release candidate (`rc`), each time resetting to a prerelease 
   number of 0. After `rc`, the prerelease suffix (e.g., `rc3`) is stripped, leaving the `major.minor.patch` version.
3. Release the next minor version (e.g., `0.1.0b1` → `0.1.0`):
   ```shell
   OLD_VERSION=$(poetry version --short)
   poetry version minor
   VERSION=$(poetry version --short)
   git commit -a -m "New minor release:  ${OLD_VERSION} → ${VERSION}"
   git push
   ```
4. Release the next major version (e.g., `0.1.0a0` → `2.0.0`):
   ```shell
   OLD_VERSION=$(poetry version --short)
   poetry version major
   VERSION=$(poetry version --short)
   git commit -a -m "New major release:  ${OLD_VERSION} → ${VERSION}"
   git push
   ```
5. Tag the current version:
   ```shell
   VERSION=$(poetry version --short)
   git tag -a -m "rlai v${VERSION}" "v${VERSION}"
   git push --follow-tags
   ```
6. Begin the next minor prerelease (e.g., `0.1.0` → `0.2.0a0`):
   ```shell
   OLD_VERSION=$(poetry version --short)
   poetry version preminor
   VERSION=$(poetry version --short)
   git commit -a -m "Next minor prerelease:  ${OLD_VERSION} → ${VERSION}"
   git push
   ```