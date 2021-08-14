from typing import Optional, Dict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from rlai.meta import rl_text
from rlai.models.sklearn import SKLearnSGD as BaseSKLearnSGD
from rlai.q_S_A.function_approximation.models import (
    FeatureExtractor,
    FunctionApproximationModel
)


@rl_text(chapter=9, page=200)
class SKLearnSGD(FunctionApproximationModel, BaseSKLearnSGD):
    """
    Extension of the base SKLearnSGD features specific to state-action modeling.
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

        FunctionApproximationModel.plot(
            self,
            feature_extractor=feature_extractor,
            policy_improvement_count=policy_improvement_count,
            num_improvement_bins=num_improvement_bins,
            render=render,
            pdf=pdf
        )

        return BaseSKLearnSGD.plot(
            self,
            render=render,
            pdf=pdf
        )

    def get_feature_action_coefficients(
            self,
            feature_extractor: FeatureExtractor
    ) -> Optional[pd.DataFrame]:
        """
        Get a pandas.DataFrame containing one row per feature and one column per action, with the cells containing the
        coefficient value of the associated feature-action pair.

        :param feature_extractor: Feature extractor used to build the model.
        :return: DataFrame (#features, #actions). The DataFrame is indexed by feature name.
        """

        # not all feature extractors return action/feature names. bail if the given one does not.
        action_feature_names = feature_extractor.get_action_feature_names()
        if action_feature_names is None:
            return None

        # model might not yet be fit (e.g., if called from jupyter notebook) and won't have a coefficients attribute in
        # such cases
        if not hasattr(self.model, 'coef_'):  # pragma no cover
            return None

        coefficients = self.model.coef_

        all_feature_names = [
            feature
            for action in action_feature_names
            for feature in action_feature_names[action]
        ]

        if 'intercept' in all_feature_names:  # pragma no cover
            raise ValueError('Feature extractors may not extract a feature named "intercept".')

        # check feature extractor names against model dimensions
        num_feature_names = len(all_feature_names)
        num_dims = coefficients.shape[0]
        if num_feature_names != num_dims:  # pragma no cover
            raise ValueError(f'Number of feature names ({num_feature_names}) does not match number of dimensions ({num_dims}).')

        # create dataframe
        df_index = all_feature_names
        if self.model.fit_intercept:
            df_index.append('intercept')

        coefficients_df = pd.DataFrame(
            index=df_index,
            columns=action_feature_names.keys()
        )

        curr_coef = 0
        for action in action_feature_names:

            feature_names = action_feature_names[action]
            num_coefs = len(feature_names)
            coefs = coefficients[curr_coef:curr_coef + num_coefs]

            for feature_name, coef in zip(feature_names, coefs):
                coefficients_df.loc[feature_name, action] = coef

            curr_coef += num_coefs

        if curr_coef != num_dims:  # pragma no cover
            raise ValueError('Failed to extract all coefficients.')

        # add intercept if we fit one
        if self.model.fit_intercept:
            coefficients_df.loc['intercept', :] = self.model.intercept_[0]

        return coefficients_df

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

        FunctionApproximationModel.__init__(self)

        BaseSKLearnSGD.__init__(
            self,
            scale_eta0_for_y=scale_eta0_for_y,
            **kwargs
        )

    def __getstate__(
            self
    ) -> Dict:
        """
        Get the state to pickle for the current instance.

        :return: State dictionary.
        """

        state = dict(self.__dict__)

        FunctionApproximationModel.deflate_state(state)
        BaseSKLearnSGD.deflate_state(state)

        return state

    def __setstate__(
            self,
            state: Dict
    ):
        """
        Set the unpickled state for the current instance.

        :param state: Unpickled state.
        """

        FunctionApproximationModel.inflate_state(state)
        BaseSKLearnSGD.inflate_state(state)

        self.__dict__ = state
