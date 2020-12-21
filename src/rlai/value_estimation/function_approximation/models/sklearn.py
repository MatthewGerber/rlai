from argparse import ArgumentParser, Namespace
from typing import Tuple, List, Optional

import numpy as np
from sklearn.linear_model import SGDRegressor

from rlai.value_estimation.function_approximation.models import FunctionApproximationModel


class SKLearnSGD(FunctionApproximationModel):

    @classmethod
    def parse_arguments(
            cls,
            args
    ) -> Tuple[Namespace, List[str]]:
        """
        Parse arguments.

        :param args: Arguments.
        :return: 2-tuple of parsed and unparsed arguments.
        """

        parsed_args, unparsed_args = super().parse_arguments(args)

        parser = ArgumentParser(allow_abbrev=False)

        # future arguments to be added here...

        parsed_args, unparsed_args = parser.parse_known_args(unparsed_args, parsed_args)

        return parsed_args, unparsed_args

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str]
    ) -> Tuple[FunctionApproximationModel, List[str]]:
        """
        Initialize a model from arguments.

        :param args: Arguments.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = cls.parse_arguments(args)

        model = SKLearnSGD(
            **vars(parsed_args)
        )

        return model, unparsed_args

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            weight: Optional[float]
    ):
        self.model.partial_fit(X=X, y=y, sample_weight=weight)

    def evaluate(
            self,
            X: np.ndarray
    ) -> np.ndarray:

        return self.model.predict(X)

    def __init__(
            self,
            **kwargs
    ):
        self.model = SGDRegressor(**kwargs)

    def __eq__(
            self,
            other
    ) -> bool:

        raise ValueError('Not implemented')

    def __ne__(
            self,
            other
    ) -> bool:

        raise ValueError('Not implemented')