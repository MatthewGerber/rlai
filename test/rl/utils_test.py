from statistics import mean

import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_allclose

from rl.utils import IncrementalSampleAverager, sample_list_item


def test_incremental_averager():

    rng = RandomState(1234)

    sample = [
        float(value)
        for value in rng.randint(0, 1000, 100)
    ]

    averager = IncrementalSampleAverager()

    for value in sample:
        averager.update(value)

    assert averager.get_value() == mean(sample)


def test_sample_list_item():

    x = [1, 2, 3]
    p = [0.1, 0.3, 0.6]

    rng = RandomState(12345)
    x_samples = [
        sample_list_item(x, p, rng)
        for _ in range(10000)
    ]

    xs, cnts = np.unique(x_samples, return_counts=True)

    x_cnt = {
        x: cnt
        for x, cnt in zip(xs, cnts)
    }

    total = sum(x_cnt.values())
    x_p = [
        x_cnt[x] / total
        for x in x
    ]

    assert_allclose(p, x_p, atol=0.01)
