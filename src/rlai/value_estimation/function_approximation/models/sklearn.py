import sys
from argparse import ArgumentParser
from typing import Tuple, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from numpy.random import RandomState
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDRegressor

from rlai.meta import rl_text
from rlai.utils import parse_arguments, IncrementalSampleAverager, StdStreamTee
from rlai.value_estimation.function_approximation.models import FunctionApproximationModel, FeatureExtractor
from rlai.value_estimation.function_approximation.models.feature_extraction import (
    StateActionInteractionFeatureExtractor
)


@rl_text(chapter=9, page=200)
class SKLearnSGD(FunctionApproximationModel):
    """
    Wrapper for the sklearn.linear_model.SGDRegressor implemented by scikit-learn.
    """

    @classmethod
    def get_argument_parser(
            cls
    ) -> ArgumentParser:
        """
        Get argument parser.

        :return: Argument parser.
        """

        parser = ArgumentParser(
            prog=f'{cls.__module__}.{cls.__name__}',
            parents=[super().get_argument_parser()],
            allow_abbrev=False,
            add_help=False
        )

        # loss
        parser.add_argument(
            '--loss',
            type=str,
            default='squared_loss',
            help='The loss function to be used. The possible values are `squared_loss`, `huber`, `epsilon_insensitive`, or `squared_epsilon_insensitive`.'
        )

        parser.add_argument(
            '--sgd-epsilon',
            type=float,
            default=0.1,
            help='Epsilon in the epsilon-insensitive loss functions.'
        )

        # regularization
        parser.add_argument(
            '--penalty',
            type=str,
            default='l2',
            help='The penalty (aka regularization term) to be used. The possible values are `l2`, `l1`, `elasticnet`.'
        )

        parser.add_argument(
            '--l1-ratio',
            type=float,
            default=0.15,
            help='The elasticnet mixing parameter (0 for pure L2 and 1 for pure L1.'
        )

        parser.add_argument(
            '--sgd-alpha',
            type=float,
            default=0.0001,
            help='Constant that multiplies the regularization term. The higher the value, the stronger the regularization. Also used to compute the learning rate when set to learning_rate is set to `optimal`.'
        )

        # learning rate (step size)
        parser.add_argument(
            '--learning-rate',
            type=str,
            default='invscaling',
            help='Learning rate schedule:  `constant`, `optimal`, `invscaling`, or `adaptive`.'
        )

        parser.add_argument(
            '--eta0',
            type=float,
            default=0.01,
            help='The initial learning rate for the `constant`, `invscaling` or `adaptive` schedules.'
        )

        parser.add_argument(
            '--power-t',
            type=float,
            default=0.25,
            help='The exponent for inverse scaling learning rate.'
        )

        parser.add_argument(
            '--scale-eta0-for-y',
            action='store_true',
            help='Scale eta0 dynamically with respect to y.'
        )

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[FunctionApproximationModel, List[str]]:
        """
        Initialize a model from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # process arguments whose names conflict with arguments used elsewhere
        setattr(parsed_args, 'alpha', parsed_args.sgd_alpha)
        del parsed_args.sgd_alpha
        setattr(parsed_args, 'epsilon', parsed_args.sgd_epsilon)
        del parsed_args.sgd_epsilon

        # instantiate model
        model = SKLearnSGD(
            random_state=random_state,
            **vars(parsed_args)
        )

        return model, unparsed_args

    def fit(
            self,
            X: Any,
            y: Any,
            weight: Optional[float]
    ):
        """
        Fit the model to a matrix of features (one row per observations) and a vector of returns.

        :param X: Feature matrix (#obs, #features)
        :param y: Outcome vector (#obs,).
        :param weight: Weight.
        """

        if self.scale_eta0_for_y:
            eta0_scalar = 1.01 ** max(np.abs(np.array(y)).max(), 1.0)
            self.model.eta0 = self.base_eta0 / eta0_scalar

        stdout_tee = StdStreamTee(sys.stdout, 20)
        sys.stdout = stdout_tee
        self.model.partial_fit(X=X, y=y, sample_weight=weight)
        sys.stdout = sys.__stdout__
        fit_line = stdout_tee.buffer[-2]
        avg_loss = float(fit_line.rsplit(' ', maxsplit=1)[1])  # example line:  Norm: 6.38, NNZs: 256, Bias: 8.932199, T: 1, Avg. loss: 0.001514

        for y_value in y:
            self.y_values.append(y_value)
            self.y_averager.update(y_value)
            self.loss_values.append(avg_loss)
            self.loss_averager.update(avg_loss)
            self.eta0_values.append(self.model.eta0)
            self.eta0_averager.update(self.model.eta0)

    def evaluate(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the model at a matrix of features (one row per observation).

        :param X: Feature matrix (#obs, #features).
        :return: Vector of outcomes from the evaluation (#obs,).
        """

        try:

            # predict values using the currently fitted model
            values = self.model.predict(X)

        # the following exception will be thrown if the model has not yet been fitted. catch and return uniform values.
        except NotFittedError:
            values = np.repeat(0.0, X.shape[0])

        return values

    def get_feature_action_coefficients(
            self,
            feature_extractor: FeatureExtractor
    ) -> Optional[pd.DataFrame]:
        """
        Get a pandas.DataFrame containing one row per feature and one column per action, with the cells containing the
        coefficient value of the associated feature-action pair.

        :param feature_extractor: Feature extractor used to build the model.
        :return: DataFrame (#features, #actions).
        """

        if isinstance(feature_extractor, StateActionInteractionFeatureExtractor):

            # get (#features, #actions) array of coefficients, along with feature names.
            num_actions = len(feature_extractor.actions)
            coefficients = self.model.coef_.reshape((-1, num_actions), order='F')
            feature_names = feature_extractor.get_feature_names()

            # add intercept if we fit one
            if self.model.fit_intercept:
                coefficients = np.append(coefficients, [np.repeat(self.model.intercept_, num_actions)], axis=0)
                feature_names.append('intercept')

            # check feature extractor names against model dimensions
            num_feature_names = len(feature_names)
            num_dims = coefficients.shape[0]
            if num_feature_names != num_dims:
                raise ValueError(f'Number of feature names ({num_feature_names}) does not match number of dimensions ({num_dims}).')

            # convert to dataframe with named columns
            coefficients = pd.DataFrame(
                data=coefficients,
                columns=[a.name for a in feature_extractor.actions]
            )

            coefficients['feature_name'] = feature_names

        else:
            raise ValueError(f'Unknown feature extractor type:  {type(feature_extractor)}')

        return coefficients

    def plot(
            self,
            feature_extractor: FeatureExtractor,
            policy_improvement_count: int,
            num_improvement_bins: Optional[int],
            do_plot: bool,
            pdf: PdfPages
    ):
        """
        Plot the model.

        :param feature_extractor: Feature extractor used to build the model.
        :param policy_improvement_count: Number of policy improvements that have been made.
        :param num_improvement_bins: Number of bins to plot.
        :param do_plot: Whether or not to plot.
        :param pdf: PDF for plots.
        """

        super().plot(
            feature_extractor=feature_extractor,
            policy_improvement_count=policy_improvement_count,
            num_improvement_bins=num_improvement_bins,
            do_plot=do_plot,
            pdf=pdf
        )

        if self.y_averager.n > 0:
            self.y_averages.append(self.y_averager.get_value())
            self.y_averager.reset()

        if self.loss_averager.n > 0:
            self.loss_averages.append(self.loss_averager.get_value())
            self.loss_averager.reset()

        if self.eta0_averager.n > 0:
            self.eta0_averages.append(self.eta0_averager.get_value())
            self.eta0_averager.reset()

        if do_plot:

            x_values = np.arange(1, len(self.y_averages) + 1)
            plt.plot(x_values, self.y_averages, label='Reward (G) obtained')
            plt.plot(x_values, self.loss_averages, label='Average model loss')
            plt.xticks(x_values)
            plt.grid()
            plt.legend(loc='upper left')
            plt.xlabel('Policy improvement iteration')

            eta0_ax = plt.twinx()
            eta0_ax.plot(x_values, self.eta0_averages, linestyle='--', label='Step size (eta0)')
            eta0_ax.legend(loc='upper right')

            if pdf is None:
                plt.show()
            else:
                pdf.savefig()

            x_values = np.arange(1, len(self.y_values) + 1)
            plt.plot(x_values, self.y_values, label='Reward (G) obtained')
            plt.plot(x_values, self.loss_values, label='Average model loss')
            plt.xlabel('Time step')
            plt.grid()
            plt.legend(loc='upper left')

            eta0_ax = plt.twinx()
            eta0_ax.plot(x_values, self.eta0_values, linestyle='--', label='Step size (eta0)')
            eta0_ax.legend(loc='upper right')

            self.y_values.clear()
            self.loss_values.clear()
            self.eta0_values.clear()

            if pdf is None:
                plt.show()
            else:
                pdf.savefig()

    def __init__(
            self,
            scale_eta0_for_y: bool,
            **kwargs
    ):
        """
        Initialize the model.

        :param scale_eta0_for_y: Whether or not to scale the value of eta0 for y.
        :param kwargs: Keyword arguments to pass to SGDRegressor.
        """

        super().__init__()

        self.scale_eta0_for_y = scale_eta0_for_y

        # verbose is required in order to capture output for plotting
        kwargs['verbose'] = 1
        self.model = SGDRegressor(**kwargs)
        self.base_eta0 = self.model.eta0
        self.y_values = []
        self.y_averager = IncrementalSampleAverager()
        self.y_averages = []
        self.loss_values = []
        self.loss_averager = IncrementalSampleAverager()
        self.loss_averages = []
        self.eta0_values = []
        self.eta0_averager = IncrementalSampleAverager()
        self.eta0_averages = []

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the model equals another.

        :param other: Other model.
        :return: True if equal and False otherwise.
        """

        return np.array_equal(self.model.coef_, other.model.coef_) and self.model.intercept_ == other.model.intercept_

    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the model does not equal another.

        :param other: Other model.
        :return: True if not equal and False otherwise.
        """

        return not (self == other)
