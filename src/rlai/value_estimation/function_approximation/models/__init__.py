import math
import threading
import warnings
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Tuple, List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from numpy.random import RandomState

from rlai.meta import rl_text
from rlai.value_estimation.function_approximation.models.feature_extraction import (
    FeatureExtractor,
    StateActionInteractionFeatureExtractor
)

MAX_PLOT_COEFFICIENTS = 10
MAX_PLOT_ACTIONS = 10


@rl_text(chapter=9, page=197)
class FunctionApproximationModel(ABC):
    """
    Function approximation model.
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
            allow_abbrev=False,
            add_help=False
        )

        parser.add_argument(
            '--help',
            action='store_true',
            help='Pass this flag to print usage and argument descriptions.'
        )

        return parser

    @classmethod
    @abstractmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[Any, List[str]]:
        """
        Initialize a model from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
        """

    @abstractmethod
    def fit(
            self,
            X: Any,
            y: Any,
            weight: Optional[float]
    ):
        """
        Fit the model to a matrix of features and a vector of returns.

        :param X: Feature matrix.
        :param y: Returns.
        :param weight: Weight.
        """

    @abstractmethod
    def evaluate(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the model at a matrix of observations.

        :param X: Observations.
        :return: Vector of evaluations.
        """

    def plot(
            self,
            feature_extractor: FeatureExtractor,
            policy_improvement_count: int,
            num_improvement_bins: Optional[int],
            render: bool,
            pdf: Optional[PdfPages]
    ) -> Optional[plt.Figure]:
        """
        Plot the model.

        :param feature_extractor: Feature extractor used to build the model.
        :param policy_improvement_count: Number of policy improvements that have been made.
        :param num_improvement_bins: Number of bins to plot.
        :param render: Whether or not to render the plot data. If False, then plot data will be updated but nothing will
        be shown.
        :param pdf: PDF for plots.
        :return: Matplotlib figure, if one was generated and not plotting to PDF.
        """

        # TODO:  update the current function to follow the ui/background thread pattern used elsewhere to surface plots
        # in the jupyter notebook.
        if threading.current_thread() != threading.main_thread():
            return

        feature_action_coefficients = self.get_feature_action_coefficients(feature_extractor)

        plot_coefficients = True

        # some models/extractors return no coefficients
        if feature_action_coefficients is None:
            plot_coefficients = False

        # some return too many
        elif feature_action_coefficients.shape[0] > MAX_PLOT_COEFFICIENTS or feature_action_coefficients.shape[1] > MAX_PLOT_ACTIONS:
            plot_coefficients = False
            warnings.warn(f'Feature-action coefficient DataFrame is too large to generate boxplots for ({feature_action_coefficients.shape}). Skipping feature-action coefficient boxplots.')

        if plot_coefficients:

            if 'n' in feature_action_coefficients.columns:  # pragma no cover
                raise ValueError('Feature extractor returned disallowed column:  n')

            if 'bin' in feature_action_coefficients.columns:  # pragma no cover
                raise ValueError('Feature extractor returned disallowed column:  bin')

            feature_action_coefficients['n'] = policy_improvement_count - 1

            if self.feature_action_coefficients is None:
                self.feature_action_coefficients = feature_action_coefficients
            else:
                self.feature_action_coefficients = self.feature_action_coefficients.append(feature_action_coefficients, ignore_index=True)

            if render:

                plt.close('all')

                if num_improvement_bins is None:
                    improvements_per_bin = 1
                    self.feature_action_coefficients['bin'] = self.feature_action_coefficients.n
                else:
                    improvements_per_bin = math.ceil(policy_improvement_count / num_improvement_bins)
                    self.feature_action_coefficients['bin'] = [
                        int(n / improvements_per_bin)
                        for n in self.feature_action_coefficients.n
                    ]

                if isinstance(feature_extractor, StateActionInteractionFeatureExtractor):

                    # set up plots
                    feature_names = self.feature_action_coefficients.feature_name.unique().tolist()
                    n_rows = len(feature_names)
                    action_names = [a.name for a in feature_extractor.actions]
                    n_cols = len(action_names)
                    fig, axs = plt.subplots(
                        nrows=n_rows,
                        ncols=n_cols,
                        sharex='all',
                        sharey='row',
                        figsize=(3 * n_cols, 3 * n_rows)
                    )

                    # plot one row per feature and one column per action, with the the plots in the array being
                    # boxplots of coefficient values.
                    for i, feature_name in enumerate(feature_names):
                        feature_df = self.feature_action_coefficients[self.feature_action_coefficients.feature_name == feature_name]
                        boxplot_axs = axs[i, :]
                        feature_df.boxplot(column=action_names, by='bin', ax=boxplot_axs)
                        boxplot_axs[0].set_ylabel(f'w({feature_name})')

                    # reset labels and titles
                    for i, row in enumerate(axs):
                        for ax in row:

                            if i < axs.shape[0] - 1:
                                ax.set_xlabel('')
                            else:
                                ax.set_xlabel('Iteration' if num_improvement_bins is None else f'Bin of {improvements_per_bin} improvement(s)')

                            if i > 0:
                                ax.set_title('')

                    fig.suptitle('Model coefficients over iterations')

                    plt.tight_layout()

                    if pdf is None:
                        plt.show(block=False)
                        return fig
                    else:
                        pdf.savefig()

                else:  # pragma no cover
                    raise ValueError(f'Unknown feature extractor type:  {type(feature_extractor)}')

    def update_plot(
            self,
            time_step_detail_iteration: Optional[int]
    ):
        """
        Update the plot of the model. Can only be called from the main thread.

        :param time_step_detail_iteration: Iteration for which to plot time-step-level detail, or None for no detail.
        Passing -1 will plot detail for the most recently completed iteration.
        """

    def get_feature_action_coefficients(
            self,
            feature_extractor: FeatureExtractor
    ) -> Optional[pd.DataFrame]:
        """
        Get a pandas.DataFrame containing one row per feature and one column per action, with the cells containing the
        coefficient value of the associated feature-action pair.

        :param feature_extractor: Feature extractor used to build the model.
        :return: DataFrame (#features, #actions), or None to omit plotting of feature-action coefficient boxplots.
        """

    def __init__(
            self
    ):
        """
        Initialize the model.
        """

        self.feature_action_coefficients: Optional[pd.DataFrame] = None

    @abstractmethod
    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the model equals another.

        :param other: Other model.
        :return: True if equal and False otherwise.
        """

    @abstractmethod
    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the model does not equal another.

        :param other: Other model.
        :return: True if not equal and False otherwise.
        """
