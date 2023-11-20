from rlai.actions import Action
from rlai.states import State


def test_ne():
    """
    Test.
    """

    s1 = State(1, [Action(1), Action(2)])
    s2 = State(2, [Action(1), Action(2)])
    assert s1 != s2

    s3 = State(1, [Action(3)])
    assert s1 == s3
