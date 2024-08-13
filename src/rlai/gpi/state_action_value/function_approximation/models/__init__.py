import math
import threading
import warnings
from abc import ABC
from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from rlai.docs import rl_text
from rlai.gpi.state_action_value.function_approximation.models.feature_extraction import StateActionFeatureExtractor
from rlai.models import FunctionApproximationModel

MAX_PLOT_COEFFICIENTS = 50
MAX_PLOT_ACTIONS = 20


@rl_text(chapter=9, page=197)
class StateActionFunctionApproximationModel(FunctionApproximationModel, ABC):
    """
    Base class for models that approximate state-action value functions.
    """

    def __init__(
            self
    ):
        """
        Initialize the model.
        """

        super().__init__()

        self.feature_action_coefficients: Optional[pd.DataFrame] = None

    def plot(
            self,
            feature_extractor: StateActionFeatureExtractor,
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
        :param render: Whether to render the plot data. If False, then plot data will be updated but nothing will
        be shown.
        :param pdf: PDF for plots.
        :return: Matplotlib figure, if one was generated and not plotting to PDF.
        """

        # TODO:  update the current function to follow the ui/background thread pattern used elsewhere to surface plots
        #   in the jupyter notebook.
        if threading.current_thread() != threading.main_thread():
            return None

        feature_action_coefficients = self.get_feature_action_coefficients(feature_extractor)

        plot_coefficients = True

        # some models/extractors return no coefficients
        if feature_action_coefficients is None:
            plot_coefficients = False

        # some return too many
        elif (
            feature_action_coefficients.shape[0] > MAX_PLOT_COEFFICIENTS or
            feature_action_coefficients.shape[1] > MAX_PLOT_ACTIONS
        ):
            plot_coefficients = False
            warnings.warn(
                f'Feature-action coefficient DataFrame is too large to generate boxplots for '
                f'({feature_action_coefficients.shape}). Skipping feature-action coefficient boxplots.'
            )

        if plot_coefficients:

            assert feature_action_coefficients is not None

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
                self.feature_action_coefficients = pd.concat(
                    [self.feature_action_coefficients, feature_action_coefficients],
                    verify_integrity=False
                )

            if render:

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

                assert isinstance(boxplot_axs, np.ndarray)

                # plot one row per feature and one column per action, with the plots in the array being boxplots of
                # coefficient values within the bin. only plot actions (columns) that have non-nan values for the
                # current feature (row).
                for i, feature_name in enumerate(feature_names):
                    feature_df = self.feature_action_coefficients[
                        self.feature_action_coefficients.feature_name == feature_name
                    ]
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

                plt.close()

        return None

    def get_feature_action_coefficients(
            self,
            feature_extractor: StateActionFeatureExtractor
    ) -> Optional[pd.DataFrame]:
        """
        Get a pandas.DataFrame containing one row per feature and one column per action, with the cells containing the
        coefficient value of the associated feature-action pair.

        :param feature_extractor: Feature extractor used to build the model.
        :return: DataFrame (#features, #actions), or None to omit plotting of feature-action coefficient boxplots. The
        DataFrame is indexed by feature name.
        """

    def __getstate__(
            self
    ) -> Dict:
        """
        Get state dictionary for pickling.

        :return: State dictionary.
        """

        # create a copy of the dictionary, so we don't modify the current object.
        state = dict(self.__dict__)

        # don't pickle the history of feature action coefficients used for plotting, as they grow unbounded during
        # training. the only known use case for saving them is to continue plotting after resumption; however, that's
        # a pretty narrow use case and isn't worth the large amount of disk space that it takes.
        state['feature_action_coefficients'] = None

        return state

    def __setstate__(
            self,
            state: Dict
    ):
        """
        Set the unpickled state for the current instance.

        :param state: Unpickled state.
        """

        self.__dict__ = state
