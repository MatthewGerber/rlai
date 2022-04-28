import numpy as np

from rlai.actions import Action, DiscretizedAction


def test_action_eq_ne():

    a1 = Action(1)
    a2 = Action(1)
    a3 = Action(2)

    assert a1 == a2 and a2 == a1 and a1 != a3 and a3 != a1


def test_action_str():

    action = Action(1, 'foo')

    assert str(action) == '1:  foo'


def test_discretized_action():

    action = DiscretizedAction(1, np.array([0.5]), 'foo')

    assert np.array_equal(action.continuous_value, np.array([0.5]))
