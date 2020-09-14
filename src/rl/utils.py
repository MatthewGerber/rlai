

class IncrementalAverager:

    def __init__(
            self
    ):
        self.average = 0.0
        self.n = 0

    def update(
            self,
            value: float
    ) -> float:

        self.n += 1.0
        self.average = self.average + (1 / self.n) * (value - self.average)

        return self.average

    def value(
            self
    ):
        return self.average
