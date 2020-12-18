from typing import Optional

from rlai.meta import rl_text
from rlai.value_estimation import ValueEstimator


@rl_text(chapter='Value Estimation', page=195)
class ApproximateValueEstimator(ValueEstimator):
    """
    Approximate value estimator.
    """

    def update(
            self,
            value: float,
            weight: Optional[float] = None
    ):
        pass

    def get_value(
            self
    ) -> float:
        pass

    def __str__(
            self
    ) -> str:
        pass
