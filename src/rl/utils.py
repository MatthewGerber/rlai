from rl.meta import rl_text


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
            value: float
    ) -> float:
        """
        Update the sample average.

        :param value: Sample value.
        :return: Updated sample average.
        """

        self.n += 1.0

        if self.alpha is None:
            step_size = 1 / self.n
        else:
            step_size = self.alpha

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
            alpha: float = None
    ):
        """
        Initialize the averager.

        :param initial_value: Initial value of the averager. Use values greater than zero to implement optimistic
        initial values, which encourages exploration in the early stages of the run.
        :param alpha: Constant step-size value. If provided, the sample average becomes a recency-weighted average with
        the weight of previous values decreasing according to `alpha^i`, where `i` is the number of time steps prior to
        the current when a previous value was obtained. If `None` is passed, then the unweighted sample average will be
        used, and every value will have the same weight.
        """

        if alpha is not None and alpha <= 0:
            raise ValueError('alpha must be > 0')

        self.initial_value = initial_value
        self.alpha = alpha
        self.average = 0.0
        self.n = initial_value

    def __str__(
            self
    ) -> str:

        return str(self.average)
