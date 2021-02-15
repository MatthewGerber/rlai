import sys
import threading
import time
from argparse import ArgumentParser
from typing import Tuple, List, Optional, Any, Dict

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
            help='Pass this flag to scale eta0 dynamically with respect to y.'
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

        # TODO:  Add more here? (e.g., put max(y) in the exponent for some base we expose)
        if self.scale_eta0_for_y:
            eta0_scalar = np.abs(np.array(y)).max()
            self.model.eta0 = self.base_eta0 / eta0_scalar

        # put tee on standard output in order to grab the loss value printed by sklearn
        stdout_tee = StdStreamTee(sys.stdout, 20, self.print_output)
        sys.stdout = stdout_tee

        # update fit
        self.model.partial_fit(X=X, y=y, sample_weight=weight)

        # reassign standard output
        sys.stdout = sys.__stdout__

        # get loss emitted by sklearn
        fit_line = stdout_tee.buffer[-2]
        if not fit_line.startswith('Norm:'):  # pragma no cover
            raise ValueError(f'Expected sklearn output to start with Norm:')

        avg_loss = float(fit_line.rsplit(' ', maxsplit=1)[1])  # example line:  Norm: 6.38, NNZs: 256, Bias: 8.932199, T: 1, Avg. loss: 0.001514

        # save y, loss, and eta0 values. each y-value is associated with the same average loss and eta0 (step size).
        with self.plot_data_lock:

            if self.plot_iteration not in self.iteration_y_values:
                self.iteration_y_values[self.plot_iteration] = []

            if self.plot_iteration not in self.iteration_loss_values:
                self.iteration_loss_values[self.plot_iteration] = []

            if self.plot_iteration not in self.iteration_eta0_values:
                self.iteration_eta0_values[self.plot_iteration] = []

            for y_value in y:
                self.iteration_y_values[self.plot_iteration].append(y_value)
                self.y_averager.update(y_value)
                self.iteration_loss_values[self.plot_iteration].append(avg_loss)
                self.loss_averager.update(avg_loss)
                self.iteration_eta0_values[self.plot_iteration].append(self.model.eta0)
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
    ) -> Optional[plt.Figure]:
        """
        Plot the model. If called from the main thread and render is True, then a new plot will be generated. If called
        from a background thread, then the data used by the plot will be updated but a plot will not be generated or
        updated. This supports a pattern in which a background thread generates new plot data, and a UI thread (e.g., in
        a Jupyter notebook) periodically calls `update_plot` to redraw the plot with the latest data.

        :param feature_extractor: Feature extractor used to build the model.
        :param policy_improvement_count: Number of policy improvements that have been made.
        :param num_improvement_bins: Number of bins to plot.
        :param render: Whether or not to render the plot data. If False, then plot data will be updated but nothing will
        be shown.
        :param pdf: PDF for plots.
        :return: Matplotlib figure, if one was generated and not plotting to PDF.
        """

        super().plot(
            feature_extractor=feature_extractor,
            policy_improvement_count=policy_improvement_count,
            num_improvement_bins=num_improvement_bins,
            render=render,
            pdf=pdf
        )

        with self.plot_data_lock:

            # collect average values for the current policy improvement iteration and reset the averagers. the
            # individual y, loss, and eta0 values have already been collected (during the calls to fit).
            if self.y_averager.n > 0:
                self.y_averages.append(self.y_averager.get_value())
                self.y_averager.reset()

            if self.loss_averager.n > 0:
                self.loss_averages.append(self.loss_averager.get_value())
                self.loss_averager.reset()

            if self.eta0_averager.n > 0:
                self.eta0_averages.append(self.eta0_averager.get_value())
                self.eta0_averager.reset()

            # sleep to let others threads (e.g., the main thread) plot if needed.
            if threading.current_thread() != threading.main_thread():
                time.sleep(0.01)

            # render the plot. the usual pattern is for this to happen only once at the start of training, and on the
            # main thread. subsequently, the main thread will call update_plot to redraw with the latest plot data
            # provided by another thread above.
            elif render:

                # noinspection PyTypeChecker
                fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(13, 6.5))

                # returns and losses per iteration
                self.iteration_ax = axs[0]
                iterations = list(range(1, len(self.y_averages) + 1))
                self.iteration_return_line, = self.iteration_ax.plot(iterations, self.y_averages, linewidth=0.75, color='darkgreen', label='Obtained')
                self.iteration_loss_line, = self.iteration_ax.plot(iterations, self.loss_averages, linewidth=0.75, color='red', label='Loss')
                self.iteration_ax.set_xlabel('Policy improvement iteration')
                self.iteration_ax.set_ylabel('Average discounted return (G)')
                self.iteration_ax.legend(loc='upper left')

                # twin-x step size
                self.iteration_eta0_ax = self.iteration_ax.twinx()
                self.iteration_eta0_line, = self.iteration_eta0_ax.plot(iterations, self.eta0_averages, linewidth=0.75, color='blue', label='Step size (eta0)')
                self.iteration_eta0_ax.set_yscale('log')
                self.iteration_eta0_ax.legend(loc='upper right')

                # plot values for all time steps of the most recent plot iteration. there might not yet be any data in
                # the current iteration, so watch out.
                self.time_step_ax = axs[1]
                y_values = self.iteration_y_values.get(self.plot_iteration, [])
                time_steps = list(range(1, len(y_values) + 1))
                self.time_step_return_line, = self.time_step_ax.plot(time_steps, y_values, linewidth=0.75, color='darkgreen', label='Obtained')
                self.time_step_loss_line, = self.time_step_ax.plot(time_steps, self.iteration_loss_values.get(self.plot_iteration, []), linewidth=0.75, color='red', label='Loss')
                self.time_step_ax.set_xlabel(f'Time step (iteration {self.plot_iteration})')
                self.iteration_ax.set_ylabel('Average discounted return (G)')
                self.time_step_ax.legend(loc='upper left')

                # twin-x step size
                self.time_step_eta0_ax = self.time_step_ax.twinx()
                self.time_step_eta0_line, = self.time_step_eta0_ax.plot(time_steps, self.iteration_eta0_values.get(self.plot_iteration, []), linewidth=0.75, color='blue', label='Step size (eta0)')
                self.time_step_eta0_ax.set_yscale('log')
                self.time_step_eta0_ax.legend(loc='upper right')

                # share y-axis scale between the two twin-x axes
                self.iteration_eta0_ax.get_shared_y_axes().join(self.iteration_eta0_ax, self.time_step_eta0_ax)

                plt.tight_layout()

                if pdf is None:
                    plt.show(block=False)
                    return fig
                else:
                    pdf.savefig()

            # move to next plot iteration
            self.plot_iteration += 1

    def update_plot(
            self,
            time_step_detail_iteration: Optional[int]
    ):
        """
        Update the plot of the model. Can only be called from the main thread.

        :param time_step_detail_iteration: Iteration for which to plot time-step-level detail, or None for no detail.
        Passing -1 will plot detail for the most recently completed iteration.
        """

        if threading.current_thread() != threading.main_thread():
            raise ValueError('Can only update plot on main thread.')

        with self.plot_data_lock:

            # plot axes will be None prior to the first call to self.plot
            if self.iteration_ax is None:
                return

            iterations = list(range(1, len(self.y_averages) + 1))

            self.iteration_return_line.set_data(iterations, self.y_averages)
            self.iteration_loss_line.set_data(iterations, self.loss_averages)
            self.iteration_ax.relim()
            self.iteration_ax.autoscale_view()

            self.iteration_eta0_line.set_data(iterations, self.eta0_averages)
            self.iteration_eta0_ax.relim()
            self.iteration_eta0_ax.autoscale_view()

            if time_step_detail_iteration is not None:

                # the current iteration is likely incomplete. use the previous iteration to ensure that we plot a
                # completed iteration. the current iteration could be complete, but the likely use case for passing -1
                # is to rapidly replot as training proceeds. in this case, the caller won't mind if they see the
                # previous iteration.
                if time_step_detail_iteration == -1:
                    time_step_detail_iteration = len(self.iteration_y_values) - 2

                if time_step_detail_iteration >= 0:

                    y_values = self.iteration_y_values[time_step_detail_iteration]
                    time_steps = list(range(1, len(y_values) + 1))

                    self.time_step_return_line.set_data(time_steps, y_values)
                    self.time_step_loss_line.set_data(time_steps, self.iteration_loss_values[time_step_detail_iteration])
                    self.time_step_ax.set_xlabel(f'Time step (iteration {time_step_detail_iteration + 1})')  # display as 1-based
                    self.time_step_ax.relim()
                    self.time_step_ax.autoscale_view()

                    self.time_step_eta0_line.set_data(time_steps, self.iteration_eta0_values[time_step_detail_iteration])
                    self.time_step_eta0_ax.relim()
                    self.time_step_eta0_ax.autoscale_view()

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

        # plotting data (update __getstate__ below when changing these attributes)
        self.iteration_y_values: Dict[int, List[float]] = {}
        self.y_averager = IncrementalSampleAverager()
        self.y_averages: List[float] = []
        self.iteration_loss_values: Dict[int, List[float]] = {}
        self.loss_averager = IncrementalSampleAverager()
        self.loss_averages: List[float] = []
        self.iteration_eta0_values: Dict[int, List[float]] = {}
        self.eta0_averager = IncrementalSampleAverager()
        self.eta0_averages: List[float] = []
        self.plot_iteration = 0  # number of policy improvement iterations that have been plotted
        self.plot_data_lock = threading.Lock()  # plotting data is read/written from multiple threads

        # plotting objects
        self.iteration_ax = None
        self.iteration_return_line = None
        self.iteration_loss_line = None
        self.iteration_eta0_ax = None
        self.iteration_eta0_line = None
        self.time_step_ax = None
        self.time_step_return_line = None
        self.time_step_loss_line = None
        self.time_step_eta0_ax = None
        self.time_step_eta0_line = None

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

    def __getstate__(
            self
    ) -> Dict:
        """
        Get the state to pickle for the current instance.

        :return: Dictionary.
        """

        state_dict = dict(self.__dict__)

        # clear other memory intensive attributes
        state_dict['plot_iteration'] = 0
        state_dict['iteration_y_values'] = {}
        state_dict['y_averager'] = IncrementalSampleAverager()
        state_dict['y_averages'] = []
        state_dict['iteration_loss_values'] = {}
        state_dict['loss_averager'] = IncrementalSampleAverager()
        state_dict['loss_averages'] = []
        state_dict['iteration_eta0_values'] = {}
        state_dict['eta0_averager'] = IncrementalSampleAverager()
        state_dict['eta0_averages'] = []

        # debatable whether plotting axes and lines should be pickled. the lines can contain a good deal of data, and
        # neither makes much sense to pickle without the above data
        state_dict['iteration_ax'] = None
        state_dict['iteration_return_line'] = None
        state_dict['iteration_loss_line'] = None
        state_dict['iteration_eta0_ax'] = None
        state_dict['iteration_eta0_line'] = None
        state_dict['time_step_ax'] = None
        state_dict['time_step_return_line'] = None
        state_dict['time_step_loss_line'] = None
        state_dict['time_step_eta0_ax'] = None
        state_dict['time_step_eta0_line'] = None

        # the plot data lock cannot be pickled
        state_dict['plot_data_lock'] = None

        return state_dict

    def __setstate__(
            self,
            state
    ):
        """
        Set the unpickled state for the current instance.

        :param state: Unpickled state.
        """

        self.__dict__ = state

        # initialize new lock, which couldn't be pickled
        self.plot_data_lock = threading.Lock()
