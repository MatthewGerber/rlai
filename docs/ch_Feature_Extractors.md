[Home](index.md) > Feature Extractors
### [rlai.core.environments.gridworld.GridworldFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/gridworld.py#L201)
```
A feature extractor for the gridworld. This extractor, being based on the `StateActionInteractionFeatureExtractor`,
    directly extracts the fully interacted state-action feature matrix. It returns numpy.ndarray feature matrices, which
    are not compatible with the Patsy formula-based interface.
```
### [rlai.core.environments.gridworld.GridworldStateFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/gridworld.py#L343)
```
A feature extractor for the gridworld. This extractor does not interact feature values with actions. Its primary use
    is in state-value estimation (e.g., for the baseline of policy gradient methods).
```
### [rlai.core.environments.gymnasium.CartpoleFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/gymnasium.py#L937)
```
A feature extractor for the Gym cartpole environment. This extractor, being based on the
    `StateActionInteractionFeatureExtractor`, directly extracts the fully interacted state-action feature matrix. It
    returns numpy.ndarray feature matrices, which are not compatible with the Patsy formula-based interface. Lastly, and
    importantly, it adds a constant term to the state-feature vector before all interactions, which results in a
    separate intercept term being present for each state segment and action combination. The function approximator
    should not add its own intercept term.
```
### [rlai.core.environments.gymnasium.ContinuousLunarLanderFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/gymnasium.py#L1488)
```
Feature extractor for the continuous lunar lander environment.
```
### [rlai.core.environments.gymnasium.ContinuousMountainCarFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/gymnasium.py#L1260)
```
Feature extractor for the continuous mountain car environment.
```
### [rlai.core.environments.gymnasium.ScaledFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/gymnasium.py#L717)
```
A feature extractor for continuous Gym environments. Extracts a scaled (standardized) version of the Gym state
    observation.
```
### [rlai.core.environments.gymnasium.SignedCodingFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/gymnasium.py#L803)
```
Signed-coding feature extractor. Forms a category from the conjunction of all state-feature signs and then places
    the continuous feature vector into its associated category. Works for all continuous-valued state spaces in Gym.
```
### [rlai.core.environments.robocode.RobocodeFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/robocode.py#L598)
```
Robocode feature extractor.
```
### [rlai.core.environments.robocode_continuous_action.RobocodeFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/core/environments/robocode_continuous_action.py#L654)
```
Robocode feature extractor.
```
### [rlai.gpi.state_action_value.function_approximation.models.feature_extraction.StateActionIdentityFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/state_action_value/function_approximation/models/feature_extraction.py#L144)
```
Simple state/action identifier extractor. Generates features named "s" and "a" for each observation. The
    interpretation of the feature values (i.e., state and action identifiers) depends on the environment. The values
    are always integers, but whether they are ordinal (ordered) or categorical (unordered) depends on the environment.
    Furthermore, it should not be assumed that the environment will provide such identifiers. They will generally be
    provided for actions (which are generally easy to enumerate up front), but this is certainly not true for states,
    which are not (easily) enumerable for all environments. All of this to say that this feature extractor is not
    generally useful. You should consider writing your own feature extractor for your environment. See
    `rlai.gpi.state_action_value.function_approximation.statistical_learning.feature_extraction.gridworld.GridworldFeatureExtractor`
    for an example.
```
### [rlai.gpi.state_action_value.function_approximation.models.feature_extraction.StateActionInteractionFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/state_action_value/function_approximation/models/feature_extraction.py#L80)
```
A feature extractor that extracts features comprising the interaction (in a statistical modeling sense) of
    state features with categorical actions. Categorical actions are coded as one-hot vectors with length equal to the
    number of possible discrete actions. To arrive at the full vector expression for a particular state-action pair, we
    first form the cartesian product of (a) the one-hot action vector and (b) the state features. Each pair in this
    product is then multiplied to arrive at the full vector expression of the state-action pair.
```
### [rlai.models.feature_extraction.FeatureScaler](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/models/feature_extraction.py#L90)
```
Base class for all feature scalers.
```
### [rlai.models.feature_extraction.NonstationaryFeatureScaler](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/models/feature_extraction.py#L186)
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
### [rlai.models.feature_extraction.OneHotCategoricalFeatureInteracter](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/models/feature_extraction.py#L311)
```
Feature interacter for one-hot encoded categorical values.
```
### [rlai.models.feature_extraction.OneHotCategory](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/models/feature_extraction.py#L362)
```
General-purpose category specification. Instances of this class are passed to
    `rlai.models.feature_extraction.OneHotCategoricalFeatureInteracter` to achieve one-hot encoding of feature vectors.
```
### [rlai.models.feature_extraction.StationaryFeatureScaler](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/models/feature_extraction.py#L123)
```
Stationary feature scaler.
```
### [rlai.state_value.function_approximation.models.feature_extraction.StateFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/state_value/function_approximation/models/feature_extraction.py#L13)
```
Feature extractor for states.
```
