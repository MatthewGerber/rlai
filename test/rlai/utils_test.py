import sys
import threading
from statistics import mean

import numpy as np
import pytest
from numpy.random import RandomState
from numpy.testing import assert_allclose

from rlai.utils import (
    IncrementalSampleAverager,
    sample_list_item,
    StdStreamTee,
    RunThreadManager,
    get_nearest_positive_definite_matrix,
    is_positive_definite
)


def test_incremental_averager():
    """
    Test.
    """

    rng = RandomState(1234)

    sample = [
        float(value)
        for value in rng.randint(0, 1000, 100)
    ]

    averager = IncrementalSampleAverager()

    for value in sample:
        averager.update(value)

    assert averager.get_value() == mean(sample)
    assert str(averager) == str(mean(sample))
    assert not (averager != averager)

    with pytest.raises(ValueError, match='Cannot pass a weight to an unweighted averager.'):
        averager.update(1.0, 1.0)

    weighted_averager = IncrementalSampleAverager(weighted=True)
    with pytest.raises(ValueError, match='The averager is weighted'):
        weighted_averager.update(1.0)

    with pytest.raises(ValueError, match='alpha must be > 0'):
        IncrementalSampleAverager(alpha=-1)

    with pytest.raises(ValueError, match='Cannot supply alpha and per-value weights.'):
        IncrementalSampleAverager(alpha=0.1, weighted=True)


def test_sample_list_item():
    """
    Test.
    """

    x = [1, 2, 3]
    p = np.array([0.1, 0.3, 0.6])

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

    with pytest.raises(ValueError, match='Expected cumulative probabilities to sum to 1'):
        sample_list_item([1, 2, 3], np.array([0.2, 0.3, 0.4]), rng)


def test_stdstream_tee():
    """
    Test.
    """

    tee = StdStreamTee(sys.stdout, 10, True)
    sys.stdout = tee  # type: ignore[assignment]

    for i in range(20):
        print(f'{i}')

    sys.stdout.flush()

    assert len(tee.buffer) == 10

    sys.stdout = sys.__stdout__


def test_run_thread_manager_initially_blocked():
    """
    Test.
    """

    run_manager = RunThreadManager(False)

    wait_return = None

    def thread_target():
        nonlocal wait_return
        wait_return = run_manager.wait(2)

    t = threading.Thread(target=thread_target)
    t.start()
    t.join()

    assert not wait_return


def test_nearest_pd():
    """
    Test.
    """

    diag = np.array([[0, 0], [0, 0]])
    np.fill_diagonal(diag, 1)
    assert np.array_equal(diag, get_nearest_positive_definite_matrix(diag))

    for i in range(10):
        for j in range(2, 100):
            A = np.random.randn(j, j)
            B = get_nearest_positive_definite_matrix(A)
            assert is_positive_definite(B)
