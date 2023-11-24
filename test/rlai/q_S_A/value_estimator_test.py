import pytest

from rlai.gpi.state_action_value.function_approximation import ApproximateStateActionValueEstimator
from rlai.gpi.state_action_value.tabular import TabularStateActionValueEstimator


# noinspection PyTypeChecker
def test_invalid_epsilon():
    """
    Test.
    """

    with pytest.raises(ValueError, match='epsilon must be >= 0'):
        TabularStateActionValueEstimator(None, -1, None)

    with pytest.raises(ValueError, match='epsilon must be >= 0'):
        ApproximateStateActionValueEstimator(None, -1, None, None, None, False, None, None)
