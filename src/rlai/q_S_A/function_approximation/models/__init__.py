import math
import threading
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Dict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from rlai.meta import rl_text
from rlai.models import FunctionApproximationModel as BaseFunctionApproximationModel
from rlai.q_S_A.function_approximation.models.feature_extraction import FeatureExtractor

MAX_PLOT_COEFFICIENTS = 50
MAX_PLOT_ACTIONS = 20


@rl_text(chapter=9, page=197)
class FunctionApproximationModel(BaseFunctionApproximationModel, ABC):
    """
    Function approximation model.
    """

    def __init__(
            self
    ):
        """
        Initialize the model.
        """

        BaseFunctionApproximationModel.__init__(self)

        self.feature_action_coefficients: Optional[pd.DataFrame] = None

    def plot(
            self,
            feature_extractor: FeatureExtractor,
            policy_improvement_count: int,
            num_improvement_bins: Optional[int],
            render: bool,
            pdf: Optional[PdfPages]
    ) -> Optional[plt.Figure]:
        """
        Plot the model's feature-action coefficients.

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
            return None

        feature_action_coefficients = self.get_feature_action_coefficients(feature_extractor)

        # some models/extractors return no coefficients...
        if isinstance(feature_action_coefficients, pd.DataFrame):

            # ...and some return too many
            if feature_action_coefficients.shape[0] > MAX_PLOT_COEFFICIENTS or \
                    feature_action_coefficients.shape[1] > MAX_PLOT_ACTIONS:
                warnings.warn(f'Feature-action coefficient DataFrame is too large to generate boxplots for ({feature_action_coefficients.shape}). Skipping feature-action coefficient boxplots.')
            else:

                if 'feature_name' in feature_action_coefficients.columns:  # pragma no cover
                    raise ValueError('Feature extractor returned disallowed column:  feature_name')

                if 'n' in feature_action_coefficients.columns:  # pragma no cover
                    raise ValueError('Feature extractor returned disallowed column:  n')

                if 'bin' in feature_action_coefficients.columns:  # pragma no cover
                    raise ValueError('Feature extractor returned disallowed column:  bin')

                # get action names before adding other columns below
                action_names = [col for col in feature_action_coefficients.columns if col != 'feature_name']

                # pull index into column, as we'll have duplicate feature names after multiple appends below
                feature_action_coefficients['feature_name'] = feature_action_coefficients.index
                feature_action_coefficients.reset_index(drop=True, inplace=True)

                # set policy improvement count for current coefficients
                feature_action_coefficients['n'] = policy_improvement_count - 1

                # append to cumulative coefficients dataframe
                if self.feature_action_coefficients is None:
                    self.feature_action_coefficients = feature_action_coefficients
                else:
                    self.feature_action_coefficients = self.feature_action_coefficients.append(
                        feature_action_coefficients,
                        ignore_index=True
                    )

                assert isinstance(self.feature_action_coefficients, pd.DataFrame)

                if render:

                    plt.close('all')

                    # assign coefficients to bins
                    if num_improvement_bins is None:
                        improvements_per_bin = 1
                        self.feature_action_coefficients['bin'] = self.feature_action_coefficients.n
                    else:
                        improvements_per_bin = math.ceil(policy_improvement_count / num_improvement_bins)
                        self.feature_action_coefficients['bin'] = [
                            int(n / improvements_per_bin)
                            for n in self.feature_action_coefficients.n
                        ]

                    # set up plots
                    feature_names = self.feature_action_coefficients.feature_name.unique().tolist()
                    n_rows = len(feature_names)
                    n_cols = len(action_names)
                    fig, boxplot_axs = plt.subplots(
                        nrows=n_rows,
                        ncols=n_cols,
                        sharex='all',
                        sharey='row',
                        figsize=(3 * n_cols, 3 * n_rows)
                    )

                    # plot one row per feature and one column per action, with the plots in the array being boxplots of
                    # coefficient values within the bin. only plot actions (columns) that have non-nan values for the
                    # current feature (row).
                    for i, feature_name in enumerate(feature_names):
                        feature_df = self.feature_action_coefficients[self.feature_action_coefficients.feature_name == feature_name]
                        for j, action_name in enumerate(action_names):
                            if feature_df[action_name].notna().sum() > 0:
                                feature_df.boxplot(column=action_name, by='bin', ax=boxplot_axs[i, j])

                    # reset labels and titles
                    for i, row in enumerate(boxplot_axs):

                        # y-label for each row is the feature name
                        boxplot_axs[i, 0].set_ylabel(f'w({feature_names[i]})')

                        for j, ax in enumerate(row):

                            # only boxplots in the final row have x-labels
                            if i < boxplot_axs.shape[0] - 1:
                                ax.set_xlabel('')
                            else:
                                ax.set_xlabel(
                                    'Iteration' if num_improvement_bins is None
                                    else f'Bin of {improvements_per_bin} improvement(s)'
                                )

                            # only boxplots in the first row have tiles (action names)
                            if i == 0:
                                ax.set_title(action_names[j])
                            else:
                                ax.set_title('')

                    fig.suptitle('Model coefficients over iterations')

                    plt.tight_layout()

                    if pdf is None:
                        plt.show(block=False)
                        return fig
                    else:
                        pdf.savefig()

        return None

    def get_feature_action_coefficients(
            self,
            feature_extractor: FeatureExtractor
    ) -> Optional[pd.DataFrame]:
        """
        Get a pandas.DataFrame containing one row per feature and one column per action, with the cells containing the
        coefficient value of the associated feature-action pair.

        :param feature_extractor: Feature extractor used to build the model.
        :return: DataFrame (#features, #actions), or None to omit plotting of feature-action coefficient boxplots. The
        DataFrame is indexed by feature name.
        """

    @abstractmethod
    def update_plot(
            self,
            time_step_detail_iteration: Optional[int]
    ):
        """
        Update the plot of the model. Can only be called from the main thread.

        :param time_step_detail_iteration: Iteration for which to plot time-step-level detail, or None for no detail.
        Passing -1 will plot detail for the most recently completed iteration.
        """

    def __getstate__(
            self
    ) -> Dict:
        """
        Get state dictionary for pickling.

        :return: State dictionary.
        """

        state = dict(self.__dict__)

        self.deflate_state(state)

        return state

    @staticmethod
    def deflate_state(
            state: Dict
    ):
        """
        Modify the state dictionary to exclude particular items.

        :param state: State dictionary.
        """

        # don't pickle the history of feature action coefficients used for plotting, as they grow unbounded during
        # training. the only known use case for saving them is to continue plotting after resumption; however, that's
        # a pretty narrow use case and isn't worth the large amount of disk space that it takes.
        state['feature_action_coefficients'] = None

    def __setstate__(
            self,
            state: Dict
    ):
        """
        Set the unpickled state for the current instance.

        :param state: Unpickled state.
        """

        self.inflate_state(state)

        self.__dict__ = state

    @staticmethod
    def inflate_state(
            state: Dict
    ):
        """
        Modify the state to include items that weren't pickled.

        :param state: Pickled state dictionary.
        """
