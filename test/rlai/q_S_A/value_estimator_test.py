import pytest

from rlai.q_S_A.function_approximation.estimators import ApproximateStateActionValueEstimator
from rlai.q_S_A.tabular.estimators import TabularStateActionValueEstimator


# noinspection PyTypeChecker
def test_invalid_epsilon():
    """
    Test.
    """

    with pytest.raises(ValueError, match='epsilon must be >= 0'):
        TabularStateActionValueEstimator(None, -1, None)

    with pytest.raises(ValueError, match='epsilon must be >= 0'):
        ApproximateStateActionValueEstimator(None, -1, None, None, None, False, None, None)
