import math
from typing import List, Any, Optional, Tuple

import numpy as np
from numpy.random import RandomState

from rlai.meta import rl_text


@rl_text(chapter=2, page=30)
class IncrementalSampleAverager:
    """
    An incremental, constant-time and -memory sample averager. Supports both decreasing (i.e., unweighted sample
    average) and constant (i.e., exponential recency-weighted average, pp. 32-33) step sizes.
    """

    def reset(
            self
    ):
        self.average = self.initial_value
        self.n = 0

    def update(
            self,
            value: float,
            weight: Optional[float] = None
    ) -> float:
        """
        Update the sample average with a new value.

        :param value: New value.
        :param weight: Weight of the value. This is a generalization of the following cases:

          * constant weight for all samples:  recency-weighted average (see `alpha` in the constructor).
          * 1 / n:  standard average.
          * else:  arbitrary weighting scheme (e.g., used for off-policy importance sampling).

        If `weighted` was True in the constructor, then a non-None value must be passed.

        :return: Updated sample average.
        """

        self.n += 1

        if self.has_alpha:
            step_size = self.alpha
        elif self.weighted:

            if weight is None:
                raise ValueError('The averager is weighted, so non-None values must be passed for weight.')

            self.cumulative_weight += weight
            step_size = weight / self.cumulative_weight

        else:
            step_size = 1 / self.n

        self.average = self.average + step_size * (value - self.average)

        return self.average

    def get_value(
            self
    ) -> float:
        """
        Get current average value.

        :return: Average.
        """

        return self.average

    def __init__(
            self,
            initial_value: float = 0.0,
            alpha: float = None,
            weighted: bool = False
    ):
        """
        Initialize the averager.

        :param initial_value: Initial value of the averager. Use values greater than zero to implement optimistic
        initial values, which encourages exploration in the early stages of the run.
        :param alpha: Constant step-size value. If provided, the sample average becomes a recency-weighted average with
        the weight of previous values decreasing according to `alpha^i`, where `i` is the number of time steps prior to
        the current when a previous value was obtained. If `None` is passed, then the unweighted sample average will be
        used, and every value will have the same weight.
        :param weighted: Whether or not per-value weights will be provided to calls to `update`. If this is True, then
        every call to `update` must provide a non-None value for `weight`.
        """

        if alpha is not None and alpha <= 0:
            raise ValueError('alpha must be > 0')

        if alpha is not None and weighted:
            raise ValueError('Cannot supply alpha and per-value weights.')

        self.initial_value = initial_value
        self.alpha = alpha
        self.has_alpha = self.alpha is not None
        self.weighted = weighted
        self.cumulative_weight = 0.0 if self.weighted else None
        self.average = initial_value
        self.n = 0

    def __str__(
            self
    ) -> str:

        return str(self.average)


def sample_list_item(
        x: List[Any],
        probs: Optional[np.ndarray],
        random_state: RandomState
) -> Any:
    """
    Sample a list item according to the items' probabilities.

    :param x: Items to sample.
    :param probs: Probabilities (must have same length as `x` and sum to 1), or None for uniform distribution.
    :param random_state: Random state.
    :return: Sampled list item.
    """

    if probs is None:
        probs = np.repeat(1 / len(x), len(x))

    cdf_y_rand = random_state.random_sample()

    cum_probs = probs.cumsum()
    final_cum_prob = cum_probs[-1]
    if abs(1.0 - final_cum_prob) > 0.0000001:
        raise ValueError(f'Expected cumulative probabilities to sum to 1, but got {final_cum_prob} instead.')

    x_i = next(
        i
        for i, cum_prob in enumerate(cum_probs)
        if cdf_y_rand < cum_prob
    )

    return x[x_i]


def check_termination_criteria(
        theta: Optional[float],
        num_iterations: Optional[int]
) -> Tuple[float, int]:
    """
    Check theta and number of iterations.

    :param theta: Theta.
    :param num_iterations: Number of iterations.
    :return: Normalized values.
    """

    # treat theta <= 0 as None, as the caller wants to ignore it.
    if theta is not None and theta <= 0:
        theta = None

    # treat num_iterations <= 0 as None, as the caller wants to ignore it.
    if num_iterations is not None and num_iterations <= 0:
        num_iterations = None

    if theta is None and num_iterations is None:
        raise ValueError('Either theta or num_iterations (or both) must be provided.')

    print(f'Starting evaluation:  theta={theta}, num_iterations={num_iterations}')

    return theta, num_iterations


def check_termination_conditions(
        delta: float,
        theta: Optional[float],
        iterations_finished: int,
        num_iterations: Optional[int]
) -> bool:
    """
    Check for termination.

    :param delta: Delta.
    :param theta: Theta.
    :param iterations_finished: Number of iterations that have been finished.
    :param num_iterations: Maximum number of iterations.
    :return: True for termination.
    """

    below_theta = theta is not None and delta < theta

    completed_num_iterations = False
    if num_iterations is not None:
        completed_num_iterations = iterations_finished >= num_iterations
        num_iterations_per_print = int(num_iterations * 0.05)
        if num_iterations_per_print > 0 and iterations_finished % num_iterations_per_print == 0:
            print(f'\tFinished {iterations_finished} iterations:  delta={delta}')

    if below_theta or completed_num_iterations:
        print(f'\tEvaluation completed:  iterations={iterations_finished}, delta={delta}\n')
        return True
    else:
        return False


def round_for_theta(
        v: float,
        theta: Optional[float]
) -> float:
    """
    Round a value based on the precision of theta.

    :param v: Value.
    :param theta: Theta.
    :return: Rounded value.
    """

    if theta is None:
        return v
    else:
        return round(v, int(abs(math.log10(theta)) - 1))
