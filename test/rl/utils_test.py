from statistics import mean

from numpy.random import RandomState

from rl.utils import OnlineSampleAverager


def test_incremental_averager():

    rng = RandomState(1234)

    sample = [
        float(value)
        for value in rng.randint(0, 1000, 100)
    ]

    averager = OnlineSampleAverager()

    for value in sample:
        averager.update(value)

    assert averager.get_value() == mean(sample)
