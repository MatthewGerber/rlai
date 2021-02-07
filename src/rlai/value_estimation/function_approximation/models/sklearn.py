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
            help='The loss function to be used.',
            choices=['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
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
            help='The penalty (aka regularization term) to be used.',
            choices=['l2', 'l1', 'elasticnet']
        )

        parser.add_argument(
            '--l1-ratio',
            type=float,
            default=0.15,
            help='The elasticnet mixing parameter (0 for pure L2 and 1 for pure L1).'
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
            help='Learning rate schedule.',
            choices=['constant', 'optimal', 'invscaling', 'adaptive']
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

        # other stuff
        parser.add_argument(
            '--verbose',
            type=int,
            default=0,
            help='Verbosity level.'
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

        # scale the step size according to y. TODO:  expose the base as a cli argument
        if self.scale_eta0_for_y:
            eta0_scalar = 1.01 ** max(np.abs(np.array(y)).max(), 1.0)
            self.model.eta0 = self.base_eta0 / eta0_scalar

        # put tee on standard output in order to grab the loss value printed by sklearn
        stdout_tee = StdStreamTee(sys.stdout, 20, self.print_output)
        sys.stdout = stdout_tee
        self.model.partial_fit(X=X, y=y, sample_weight=weight)
        sys.stdout = sys.__stdout__

        fit_line = stdout_tee.buffer[-2]
        if not fit_line.startswith('Norm:'):  # pragma no cover
            raise ValueError(f'Expected sklearn output to start with Norm:')

        avg_loss = float(fit_line.rsplit(' ', maxsplit=1)[1])  # example line:  Norm: 6.38, NNZs: 256, Bias: 8.932199, T: 1, Avg. loss: 0.001514

        # save y values. each is associated with the same average loss and eta0 (step size).
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

            # not all feature extractors return names. bail if the given one does not.
            feature_names = feature_extractor.get_feature_names()
            if feature_names is None:
                return None

            # get (#features, #actions) array of coefficients, along with feature names.
            num_actions = len(feature_extractor.actions)
            coefficients = self.model.coef_.reshape((-1, num_actions), order='F')

            # add intercept if we fit one
            if self.model.fit_intercept:
                coefficients = np.append(coefficients, [np.repeat(self.model.intercept_, num_actions)], axis=0)
                feature_names.append('intercept')

            # check feature extractor names against model dimensions
            num_feature_names = len(feature_names)
            num_dims = coefficients.shape[0]
            if num_feature_names != num_dims:  # pragma no cover
                raise ValueError(f'Number of feature names ({num_feature_names}) does not match number of dimensions ({num_dims}).')

            # convert to dataframe with named columns
            coefficients = pd.DataFrame(
                data=coefficients,
                columns=[a.name for a in feature_extractor.actions]
            )

            coefficients['feature_name'] = feature_names

        else:  # pragma no cover
            raise ValueError(f'Unknown feature extractor type:  {type(feature_extractor)}')

        return coefficients

    def plot(
            self,
            feature_extractor: FeatureExtractor,
            policy_improvement_count: int,
            num_improvement_bins: Optional[int],
            render: bool,
            pdf: Optional[PdfPages]
    ):
        """
        Plot the model.

        :param feature_extractor: Feature extractor used to build the model.
        :param policy_improvement_count: Number of policy improvements that have been made.
        :param num_improvement_bins: Number of bins to plot.
        :param render: Whether or not to render the plot data. If False, then plot data will be updated but nothing will
        be shown.
        :param pdf: PDF for plots.
        """

        super().plot(
            feature_extractor=feature_extractor,
            policy_improvement_count=policy_improvement_count,
            num_improvement_bins=num_improvement_bins,
            render=render,
            pdf=pdf
        )

        # collect average values for the current policy improvement iteration and reset the averagers. there's no need
        # to collect values for the time steps, as these are collected during the calls to fit.
        if self.y_averager.n > 0:
            self.y_averages.append(self.y_averager.get_value())
            self.y_averager.reset()

        if self.loss_averager.n > 0:
            self.loss_averages.append(self.loss_averager.get_value())
            self.loss_averager.reset()

        if self.eta0_averager.n > 0:
            self.eta0_averages.append(self.eta0_averager.get_value())
            self.eta0_averager.reset()

        if render:

            # noinspection PyTypeChecker
            fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 7.5))

            # plot values for all improvement iterations since the previous rendering
            ax = axs[0]
            num_plot_iterations = len(self.y_averages)
            x_values = np.arange(self.plot_start_iteration, self.plot_start_iteration + num_plot_iterations)
            ax.plot(x_values, self.y_averages, color='red', label='Reward (G) obtained')
            ax.plot(x_values, self.loss_averages, color='mediumpurple', label='Average model loss')
            ax.legend(loc='upper left')
            ax.set_xlabel('Policy improvement iteration')

            eta0_ax = ax.twinx()
            eta0_ax.plot(x_values, self.eta0_averages, linestyle='--', label='Step size (eta0)')
            eta0_ax.legend(loc='upper right')

            self.y_averages.clear()
            self.loss_averages.clear()
            self.eta0_averages.clear()
            self.plot_start_iteration += num_plot_iterations

            # plot values for all time steps since the previous rendering
            ax = axs[1]
            num_plot_time_steps = len(self.y_values)
            x_values = np.arange(self.plot_start_time_step, self.plot_start_time_step + num_plot_time_steps)
            ax.plot(x_values, self.y_values, color='red', label='Reward (G) obtained')
            ax.plot(x_values, self.loss_values, color='mediumpurple', label='Average model loss')
            ax.set_xlabel('Time step')
            ax.legend(loc='upper left')

            eta0_ax = ax.twinx()
            eta0_ax.plot(x_values, self.eta0_values, linestyle='--', label='Step size (eta0)')
            eta0_ax.legend(loc='upper right')

            self.y_values.clear()
            self.loss_values.clear()
            self.eta0_values.clear()
            self.plot_start_time_step += num_plot_time_steps

            if pdf is None:
                plt.show(block=False)
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

        # verbose is required in order to capture standard output for plotting. if a verbosity level is not passed or
        # passed as 0, then set flag indicating that we should not print captured output back to stdout; otherwise,
        # print captured output back to stdout as the caller expects.
        passed_verbose = kwargs.get('verbose', 0)
        kwargs['verbose'] = 1
        self.print_output = passed_verbose != 0

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
        self.plot_start_iteration = 1
        self.plot_start_time_step = 1

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
