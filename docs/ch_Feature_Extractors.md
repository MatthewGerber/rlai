[Home](index.md) > Feature Extractors
### [rlai.core.environments.gridworld.GridworldFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/gridworld.py#L203)
```
A feature extractor for the gridworld. This extractor, being based on the `StateActionInteractionFeatureExtractor`,
    directly extracts the fully interacted state-action feature matrix. It returns numpy.ndarray feature matrices, which
    are not compatible with the Patsy formula-based interface.
```
### [rlai.core.environments.gridworld.GridworldStateFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/gridworld.py#L329)
```
A feature extractor for the gridworld. This extractor does not interact feature values with actions. Its primary use
    is in state-value estimation (e.g., for the baseline of policy gradient methods).
```
### [rlai.core.environments.openai_gym.CartpoleFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/openai_gym.py#L625)
```
A feature extractor for the OpenAI cartpole environment. This extractor, being based on the
    `StateActionInteractionFeatureExtractor`, directly extracts the fully interacted state-action feature matrix. It
    returns numpy.ndarray feature matrices, which are not compatible with the Patsy formula-based interface.
```
### [rlai.core.environments.openai_gym.ContinuousFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/openai_gym.py#L765)
```
A feature extractor for continuous OpenAI environments.
```
### [rlai.core.environments.openai_gym.ContinuousLunarLanderFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/openai_gym.py#L901)
```
Feature extractor for the continuous lunar lander.
```
### [rlai.core.environments.openai_gym.ContinuousMountainCarFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/openai_gym.py#L974)
```
Feature extractor for the continuous lunar lander.
```
### [rlai.core.environments.openai_gym.SignedCodingFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/openai_gym.py#L848)
```
Signed-coding feature extractor. Forms a category from the conjunction of all state-feature signs and then places
    the continuous feature vector into its associated category.
```
### [rlai.core.environments.robocode.RobocodeFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/robocode.py#L592)
```
Robocode feature extractor.
```
### [rlai.core.environments.robocode_continuous_action.RobocodeFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/environments/robocode_continuous_action.py#L648)
```
Robocode feature extractor.
```
### [rlai.models.feature_extraction.NonstationaryFeatureScaler](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/models/feature_extraction.py#L66)
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
### [rlai.models.feature_extraction.OneHotCategoricalFeatureInteracter](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/models/feature_extraction.py#L167)
```
Feature interacter for one-hot encoded categorical values.
```
### [rlai.models.feature_extraction.OneHotCategory](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/models/feature_extraction.py#L217)
```
General-purpose category specification. Instances of this class are passed to
    `rlai.models.feature_extraction.OneHotCategoricalFeatureInteracter` to achieve one-hot encoding of feature vectors.
```
### [rlai.q_S_A.function_approximation.models.feature_extraction.StateActionIdentityFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/q_S_A/function_approximation/models/feature_extraction.py#L149)
```
Simple state/action identifier extractor. Generates features named "s" and "a" for each observation. The
    interpretation of the feature values (i.e., state and action identifiers) depends on the environment. The values
    are always integers, but whether they are ordinal (ordered) or categorical (unordered) depends on the environment.
    Furthermore, it should not be assumed that the environment will provide such identifiers. They will generally be
    provided for actions (which are generally easy to enumerate up front), but this is certainly not true for states,
    which are not (easily) enumerable for all environments. All of this to say that this feature extractor is not
    generally useful. You should consider writing your own feature extractor for your environment. See
    `rlai.q_S_A.function_approximation.statistical_learning.feature_extraction.gridworld.GridworldFeatureExtractor`
    for an example.
```
### [rlai.q_S_A.function_approximation.models.feature_extraction.StateActionInteractionFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/q_S_A/function_approximation/models/feature_extraction.py#L85)
```
A feature extractor that extracts features comprising the interaction (in a statistical modeling sense) of
    state features with categorical actions. Categorical actions are coded as one-hot vectors with length equal to the
    number of possible discrete actions. To arrive at the full vector expression for a particular state-action pair, we
    first form the cartesian product of (a) the one-hot action vector and (b) the state features. Each pair in this
    product is then multiplied to arrive at the full vector expression of the state-action pair.
```
### [rlai.state_value.function_approximation.models.feature_extraction.StateFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/v_S/function_approximation/models/feature_extraction.py#L11)
```
Feature extractor for states.
```
