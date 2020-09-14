from statistics import mean

from numpy.random import RandomState

from rl.utils import IncrementalAverager


def test_incremental_averager():

    rng = RandomState(1234)
    values = [
        float(value)
        for value in rng.randint(0, 1000, 100)
    ]

    averager = IncrementalAverager()

    for value in values:
        averager.update(value)

    assert averager.value() == mean(values)
