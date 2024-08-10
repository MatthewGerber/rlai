import pytest

from rlai.gpi.state_action_value.function_approximation import ApproximateStateActionValueEstimator


# noinspection PyTypeChecker
def test_invalid_epsilon():
    """
    Test.
    """

    with pytest.raises(ValueError, match='epsilon must be >= 0'):
        ApproximateStateActionValueEstimator(
            None, -1, None, None, None, False,  # type: ignore
            None, None
        )
