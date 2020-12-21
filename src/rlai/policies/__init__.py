from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np

from rlai.actions import Action
from rlai.states.mdp import MdpState


class Policy(ABC):

    @abstractmethod
    def get_state_i(
            self,
            state_descriptor: Union[str, np.ndarray]
    ) -> int:
        pass

    @abstractmethod
    def __getitem__(
            self,
            state: MdpState
    ) -> Dict[Action, float]:
        pass
