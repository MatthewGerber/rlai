import numpy as np
from numpy.random import RandomState

from rlai.core import Monitor


def test_monitor():
    """
    Test.
    """

    T = 100

    monitor = Monitor()

    rng = RandomState(12345)

    actions = rng.randint(0, 2, T).tolist()
    optimal_actions = rng.randint(0, 2, T).tolist()
    rewards = rng.random(T).tolist()

    for t in range(T):

        monitor.report(
            t=t,
            agent_action=actions[t],
            optimal_action=optimal_actions[t],
            action_reward=rewards[t]
        )

    assert np.array_equal(
        [
            monitor.t_count_optimal_action[t]
            for t in sorted(monitor.t_count_optimal_action)
        ],
        [
            1 if action == optimal else 0
            for action, optimal in zip(actions, optimal_actions)
        ]
    )

    assert np.array_equal(
        [
            monitor.t_average_cumulative_reward[t].get_value()
            for t in sorted(monitor.t_average_cumulative_reward)
        ],
        np.cumsum(rewards)
    )
