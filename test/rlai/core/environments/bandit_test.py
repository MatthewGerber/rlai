from numpy.random import RandomState

from rlai.core.environments.bandit import Arm


def test_arm():
    """
    Test.
    """

    random = RandomState()
    arm = Arm(1, 0.0, 1.0, random)

    assert str(arm) == 'Mean:  0.0, Variance:  1.0'
