from rlai.rewards import Reward


def test_reward_overrides():

    reward = Reward(1, 2)
    reward2 = Reward(1, 2)

    assert str(reward) == f'Reward {1}: {2}'
    assert not (reward != reward2)
