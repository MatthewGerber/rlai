from enum import Enum, auto


class PolicyImprovementEvent(Enum):
    """
    Events that can trigger a policy improvement.
    """

    # We finished one iteration of evaluation (over 1 or more episodes).
    FINISHED_EVALUATION = auto()

    # We updated a value estimate.
    UPDATED_VALUE_ESTIMATE = auto()

    # We're making the policy greedy.
    MAKING_POLICY_GREEDY = auto()
