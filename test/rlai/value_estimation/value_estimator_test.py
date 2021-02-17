import pytest

from rlai.value_estimation.function_approximation.estimators import ApproximateStateActionValueEstimator
from rlai.value_estimation.tabular import TabularStateActionValueEstimator


# noinspection PyTypeChecker
def test_invalid_epsilon():

    with pytest.raises(ValueError, match='epsilon must be >= 0'):
        TabularStateActionValueEstimator(None, -1, None)

    with pytest.raises(ValueError, match='epsilon must be >= 0'):
        ApproximateStateActionValueEstimator(None, -1, None, None, None, False, None, None)
