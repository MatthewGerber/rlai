import pickle
import shlex
import tempfile

from rlai.runners import top_level, trainer


def test_train():
    """
    Test.
    """

    checkpoint_path_top_level, agent_path_top_level = top_level.run(shlex.split(f'train --random-seed 12345 --agent rlai.gpi.state_action_value.ActionValueMdpAgent --gamma 1 --environment rlai.core.environments.gridworld.Gridworld --id example_4_1 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 5 --epsilon 0.01 --q-S-A rlai.gpi.state_action_value.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 10 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))
    with open(agent_path_top_level, 'rb') as f:
        agent_top_level = pickle.load(f)

    checkpoint_path_train, agent_path_train = trainer.run(shlex.split(f'--random-seed 12345 --agent rlai.gpi.state_action_value.ActionValueMdpAgent --gamma 1 --environment rlai.core.environments.gridworld.Gridworld --id example_4_1 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 5 --epsilon 0.01 --q-S-A rlai.gpi.state_action_value.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 10 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))
    with open(agent_path_train, 'rb') as f:
        agent_train = pickle.load(f)

    assert agent_top_level.pi == agent_train.pi


def test_help():
    """
    Test.
    """

    top_level.run(shlex.split('help rlai.gpi.state_action_value.ActionValueMdpAgent'))
    top_level.run(shlex.split('help rlai.policy_gradient.ParameterizedMdpAgent'))
    top_level.run(shlex.split('help rlai.core.StochasticMdpAgentXXXX'))
    top_level.run(shlex.split('help'))
