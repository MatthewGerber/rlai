from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from rlai.actions import Action
from rlai.meta import rl_text
from rlai.states.mdp import MdpState


@rl_text(chapter=9, page=197)
class FeatureExtractor(ABC):
    """
    Feature extractor.
    """

    @abstractmethod
    def extract(
            self,
            state: MdpState,
            actions: List[Action]
    ) -> pd.DataFrame:
        """
        Extract features from a state and actions.

        :param state: State.
        :param actions: Actions.
        :return: DataFrame with one row per action and one column per feature.
        """
        pass


class StateActionIdentityFeatureExtractor(FeatureExtractor):
    """
    Simple state/action identifier extractor. Generates features named "s" and "a" for each observation. The
    interpretation of the feature values (i.e., state and action identifiers) depends on the environment. The values
    are always integers, but whether they are ordinal (ordered) or categorical (unordered) depends on the environment.
    Furthermore, it should not be assumed that the environment will provide such identifiers. They will generally be
    provided for actions (which are generally easy to enumerate up front), but this is certainly not true for states,
    which are not (easily) enumerable for all environments. All of this to say that this feature extractor is not
    generally useful. You should consider writing your own feature extractor for your environment.
    """

    def extract(
            self,
            state: MdpState,
            actions: List[Action]
    ) -> pd.DataFrame:
        """
        Extract discrete state and action identifiers.

        :param state: State.
        :param actions: Actions.
        :return: DataFrame with one row per action and two columns (one for the state and one for the action).
        """

        return pd.DataFrame([
            (state.i, action.i)
            for action in actions
        ], columns=['s', 'a'])
