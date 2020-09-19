from numpy.random.mtrand import RandomState
import numpy as np
from rl.runners.monitor import Monitor


def test_monitor():

    T = 100

    monitor = Monitor(
        T=T
    )

    rng = RandomState(12345)

    actions = rng.randint(0, 2, T)
    optimal_actions = rng.randint(0, 2, T)
    rewards = rng.random(T)

    for t in range(T):

        monitor.report(
            t=t,
            agent_action=actions[t],
            optimal_action=optimal_actions[t],
            action_reward=rewards[t]
        )

    assert np.array_equal(monitor.t_count_optimal_action, [
        1 if action == optimal else 0
        for action, optimal in zip(actions, optimal_actions)
    ])
