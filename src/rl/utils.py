

class OnlineSampleAverager:
    """
    An online, constant-time and -memory sample averager.
    """

    def __init__(
            self
    ):
        self.average = 0.0
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
        self.average = self.average + (1 / self.n) * (value - self.average)

        return self.average

    def get_value(
            self
    ) -> float:
        """
        Get current average value.
        :return: Average.
        """

        return self.average
