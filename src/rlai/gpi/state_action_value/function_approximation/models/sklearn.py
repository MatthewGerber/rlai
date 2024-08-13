from typing import Optional, Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from numpy.random import RandomState

from rlai.docs import rl_text
from rlai.gpi.state_action_value.function_approximation.models import (
    StateActionFeatureExtractor,
    StateActionFunctionApproximationModel
)
from rlai.models import FunctionApproximationModel
from rlai.models.sklearn import SKLearnSGD as SKLearnSGDRegressor


@rl_text(chapter=9, page=200)
class SKLearnSGD(StateActionFunctionApproximationModel):
    """
    State-action value modeler based on the SKLearnSGD algorithm.
    """

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState,
            fit_intercept: bool
    ) -> Tuple[FunctionApproximationModel, List[str]]:
        """
        Initialize a model from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param fit_intercept: Whether to fit an intercept term.
        :return: 2-tuple of a model and a list of unparsed arguments.
        """

        sklearn_sgd, unparsed_args = SKLearnSGDRegressor.init_from_arguments(
            args=args,
            random_state=random_state,
            fit_intercept=fit_intercept
        )

        assert isinstance(sklearn_sgd, SKLearnSGDRegressor)

        state_action_sklearn_sgd = cls(
            sklearn_sgd=sklearn_sgd
        )

        return state_action_sklearn_sgd, unparsed_args

    def fit(
            self,
            feature_matrix: Any,
            outcomes: Any,
            weights: Optional[np.ndarray]
    ):
        """
        Fit the model to a matrix of feature vectors and a vector of outcomes.

        :param feature_matrix: Feature matrix (#obs, #features).
        :param outcomes: Outcome vector (#obs,).
        :param weights: Weights (#obs,).
        """

        self.sklearn_sgd.fit(feature_matrix, outcomes, weights)

    def evaluate(
            self,
            feature_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the model at a matrix of features.

        :param feature_matrix: Feature matrix (#obs, #features).
        :return: Vector of outcomes from the evaluation (#obs,).
        """

        return self.sklearn_sgd.evaluate(feature_matrix)

    def plot(
            self,
            feature_extractor: StateActionFeatureExtractor,
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
        :param render: Whether to render the plot data. If False, then plot data will be updated but nothing will
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

        return self.sklearn_sgd.plot(
            render=render,
            pdf=pdf
        )

    def get_feature_action_coefficients(
            self,
            feature_extractor: StateActionFeatureExtractor
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
        if not hasattr(self.sklearn_sgd.model, 'coef_'):  # pragma no cover
            return None

        coefficients = self.sklearn_sgd.model.coef_

        all_feature_names = [
            feature
            for action in action_feature_names
            for feature in action_feature_names[action]
        ]

        if 'intercept' in all_feature_names and self.sklearn_sgd.model.fit_intercept:  # pragma no cover
            raise ValueError(
                'Feature extractors may not extract a feature named "intercept" if the SKLearnSGD model fits an '
                'intercept. The names clash.'
            )

        # check feature extractor names against model dimensions
        num_feature_names = len(all_feature_names)
        num_dims = coefficients.shape[0]
        if num_feature_names != num_dims:  # pragma no cover
            raise ValueError(
                f'Number of feature names ({num_feature_names}) does not match number of dimensions ({num_dims}).'
            )

        # create dataframe
        df_index = all_feature_names
        if self.sklearn_sgd.model.fit_intercept:
            df_index.append('intercept')

        coefficients_df = pd.DataFrame(
            index=df_index,
            columns=list(action_feature_names.keys())
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
        if self.sklearn_sgd.model.fit_intercept:
            coefficients_df.loc['intercept', :] = self.sklearn_sgd.model.intercept_[0]

        return coefficients_df

    def __init__(
            self,
            sklearn_sgd: SKLearnSGDRegressor
    ):
        """
        Initialize the model.

        :param sklearn_sgd: SKLearnSGD instance.
        """

        super().__init__()

        self.sklearn_sgd = sklearn_sgd

    def __eq__(
            self,
            other: object
    ) -> bool:
        """
        Check equality.

        :param other: Other object.
        :return: True if equal.
        """

        if not isinstance(other, SKLearnSGD):
            raise ValueError(f'Expected a {SKLearnSGD}')

        return self.sklearn_sgd.__eq__(other.sklearn_sgd)

    def __ne__(
            self,
            other: object
    ) -> bool:
        """
        Check inequality.

        :param other: Other object.
        :return: True if not equal.
        """

        if not isinstance(other, SKLearnSGD):
            raise ValueError(f'Expected a {SKLearnSGD}')

        return self.sklearn_sgd.__ne__(other.sklearn_sgd)
