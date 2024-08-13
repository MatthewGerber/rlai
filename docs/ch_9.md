[Home](index.md) > Chapter 9:  On-policy Prediction with Approximation
### [rlai.gpi.state_action_value.function_approximation.models.StateActionFunctionApproximationModel](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/state_action_value/function_approximation/models/__init__.py#L21)
```
Base class for models that approximate state-action value functions.
```
### [rlai.gpi.state_action_value.function_approximation.models.feature_extraction.StateActionFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/state_action_value/function_approximation/models/feature_extraction.py#L16)
```
Feature extractor.
```
### [rlai.models.FunctionApproximationModel](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/models/__init__.py#L13)
```
Base class for models that approximate functions.
```
### [rlai.models.feature_extraction.FeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/models/feature_extraction.py#L16)
```
Base feature extractor for all others. This class does not define any extraction function, since the signature of
    such a function depends on what, conceptually, we're extracting features from. The definition of this signature is
    deferred to inheriting classes that are closer to their conceptual extraction targets.
```
### [rlai.state_value.function_approximation.models.StateFunctionApproximationModel](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/state_value/function_approximation/models/__init__.py#L10)
```
Base class for models that approximate state-action value functions.
```
### [rlai.state_value.function_approximation.models.sklearn.SKLearnSGD](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/state_value/function_approximation/models/sklearn.py#L13)
```
State-action value modeler based on the SKLearnSGD algorithm.
```
### [rlai.gpi.state_action_value.function_approximation.models.sklearn.SKLearnSGD](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/gpi/state_action_value/function_approximation/models/sklearn.py#L19)
```
State-action value modeler based on the SKLearnSGD algorithm.
```
### [rlai.models.sklearn.SKLearnSGD](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/models/sklearn.py#L20)
```
Wrapper for the sklearn.linear_model.SGDRegressor implemented by scikit-learn.
```
