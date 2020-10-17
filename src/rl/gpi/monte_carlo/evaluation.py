from typing import Tuple, Dict

from rl.agents.mdp import MdpAgent
from rl.meta import rl_text
from rl.states.mdp import MdpState


@rl_text(chapter=5, page=92)
def evaluate_v_pi(
        agent: MdpAgent,
        num_episodes: int
) -> Tuple[Dict[MdpState, float], float]:
    pass
