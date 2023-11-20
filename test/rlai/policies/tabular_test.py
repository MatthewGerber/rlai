import numpy as np
import pytest

from rlai.policies.tabular import TabularPolicy


def test_invalid_get_state_i():
    """
    Test.
    """

    policy = TabularPolicy(None, None)

    with pytest.raises(ValueError, match='Attempted to discretize a continuous state without a resolution.'):
        policy.get_state_i(np.array([[1, 2, 3]]))

    with pytest.raises(ValueError, match=f'Unknown state space type:  {type(3)}'):
        # noinspection PyTypeChecker
        policy.get_state_i(3)


def test_policy_not_equal():
    """
    Test.
    """

    policy_1 = TabularPolicy(None, None)
    policy_2 = TabularPolicy(None, None)

    assert not (policy_1 != policy_2)
