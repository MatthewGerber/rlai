# Feature Extractors
### [rlai.environments.gridworld.GridworldFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/gridworld.py#L201)
```
A feature extractor for the gridworld. This extractor, being based on the `StateActionInteractionFeatureExtractor`,
    directly extracts the fully interacted state-action feature matrix. It returns numpy.ndarray feature matrices, which
    are not compatible with the Patsy formula-based interface.
```
### [rlai.environments.openai_gym.CartpoleFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/openai_gym.py#L373)
```
A feature extractor for the OpenAI cartpole environment. This extractor, being based on the
    `StateActionInteractionFeatureExtractor`, directly extracts the fully interacted state-action feature matrix. It
    returns numpy.ndarray feature matrices, which are not compatible with the Patsy formula-based interface.
```
### [rlai.environments.robocode.RobocodeFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/robocode.py#L524)
```
Robocode feature extractor.
```
### [rlai.value_estimation.function_approximation.models.feature_extraction.NonstationaryFeatureScaler](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/value_estimation/function_approximation/models/feature_extraction.py#L322)
```
It is common for function approximation models to require some sort of state-feature scaling in order to converge
    upon optimal solutions. For example, in stochastic gradient descent, the use of state features with different scales
    can cause weight updates to increase loss depending on the step size and gradients of the loss function with respect
    to the weights. A common approach to scaling weights is standardization, and scikit-learn supports this with the
    StandardScaler. However, the StandardScaler is intended for use with stationary state-feature distributions, whereas
    the state-feature distributions in RL tasks can be nonstationary (e.g., a cartpole agent that moves through distinct
    state-feature distributions over the course of learning). This class provides a simple extension of the
    StandardScaler to address nonstationary state-feature scaling. It refits the scaler periodically using the most
    recent state-feature observations. Furthermore, it assigns weights to these observations that decay exponentially
    with the observation age.
```
### [rlai.value_estimation.function_approximation.models.feature_extraction.OneHotCategoricalFeatureInteracter](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/value_estimation/function_approximation/models/feature_extraction.py#L272)
```
Feature interacter for one-hot encoded categorical values.
```
### [rlai.value_estimation.function_approximation.models.feature_extraction.StateActionIdentityFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/value_estimation/function_approximation/models/feature_extraction.py#L180)
```
Simple state/action identifier extractor. Generates features named "s" and "a" for each observation. The
    interpretation of the feature values (i.e., state and action identifiers) depends on the environment. The values
    are always integers, but whether they are ordinal (ordered) or categorical (unordered) depends on the environment.
    Furthermore, it should not be assumed that the environment will provide such identifiers. They will generally be
    provided for actions (which are generally easy to enumerate up front), but this is certainly not true for states,
    which are not (easily) enumerable for all environments. All of this to say that this feature extractor is not
    generally useful. You should consider writing your own feature extractor for your environment. See
    `rlai.value_estimation.function_approximation.statistical_learning.feature_extraction.gridworld.GridworldFeatureExtractor`
    for an example.
```
### [rlai.value_estimation.function_approximation.models.feature_extraction.StateActionInteractionFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/value_estimation/function_approximation/models/feature_extraction.py#L116)
```
A feature extractor that extracts features comprising the interaction (in a statistical modeling sense) of
    state features with categorical actions. Categorical actions are coded as one-hot vectors with length equal to the
    number of possible discrete actions. To arrive at the full vector expression for a particular state-action pair, we
    first form the cartesian product of (a) the one-hot action vector and (b) the state features. Each pair in this
    product is then multiplied to arrive at the full vector expression of the state-action pair.
```
