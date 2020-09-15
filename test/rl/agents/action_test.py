from rl.agents.action import Action


def test_action_eq_ne():

    a1 = Action(1)
    a2 = Action(1)
    a3 = Action(2)

    assert a1 == a2 and a2 == a1 and a1 != a3 and a3 != a1
