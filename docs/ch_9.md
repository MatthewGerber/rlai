# Chapter 9:  On-policy Prediction with Approximation
### [rlai.value_estimation.function_approximation.statistical_learning.FunctionApproximationModel](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/value_estimation/function_approximation/statistical_learning.py#L11)
```
Function approximation model.
```
### [rlai.value_estimation.function_approximation.statistical_learning.feature_extraction.FeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/value_estimation/function_approximation/statistical_learning/feature_extraction.py#L13)
```
Feature extractor.
```
### [rlai.value_estimation.function_approximation.statistical_learning.feature_extraction.StateActionIdentityFeatureExtractor](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/value_estimation/function_approximation/statistical_learning/feature_extraction.py#L34)
```
Simple state/action identifier extractor. Generates features named "s" and "a" for each observation. The
    interpretation of the feature values (i.e., state and action identifiers) depends on the environment. The values
    are always integers, but whether they are ordinal (ordered) or categorical (unordered) depends on the environment.
    Furthermore, it should not be assumed that the environment will provide such identifiers. They will generally be
    provided for actions (which are generally easy to enumerate up front), but this is certainly not true for states,
    which are not (easily) enumerable for all environments. All of this to say that this feature extractor is not
    generally useful. You should consider writing your own feature extractor for your environment.
```
### [rlai.value_estimation.function_approximation.statistical_learning.sklearn.SKLearnSGD](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/value_estimation/function_approximation/statistical_learning/sklearn.py#L13)
```
Wrapper for the sklearn.linear_model.SGDRegressor implemented by scikit-learn.
```
