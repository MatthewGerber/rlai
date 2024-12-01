import os
import pickle
import shlex
import tempfile
from typing import Dict

from rlai.gpi.monte_carlo.iteration import iterate_value_q_pi
from rlai.gpi.utils import resume_from_checkpoint
from rlai.runners.trainer import run
from test.rlai.utils import start_virtual_display_if_headless


def test_resume_from_checkpoint():
    """
    Test.
    """

    start_virtual_display_if_headless()

    def resume_args_mutator(
            resume_args: Dict
    ):
        print(f'Called mutator:  {len(resume_args)} resume arguments.')

    def train_function_args_callback(
            args: Dict
    ):
        print(f'Called callback:  {len(args)} resume arguments.')

    run_args = (
        '--random-seed 12345 --agent rlai.gpi.state_action_value.ActionValueMdpAgent '
        '--continuous-state-discretization-resolution 0.005 --gamma 0.95 '
        '--environment rlai.core.environments.gymnasium.Gym --gym-id CartPole-v1 --render-every-nth-episode 2 '
        '--train-function rlai.gpi.monte_carlo.iteration.iterate_value_q_pi --num-improvements 2 '
        '--num-episodes-per-improvement 2 --update-upon-every-visit True --epsilon 0.2 '
        '--q-S-A rlai.gpi.state_action_value.tabular.TabularStateActionValueEstimator '
        '--make-final-policy-greedy False --num-improvements-per-plot 2 --num-improvements-per-checkpoint 2 '
        f'--checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} '
        f'--save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'
    )
    checkpoint_path, agent_path = run(
        args=shlex.split(run_args),
        train_function_args_callback=train_function_args_callback
    )

    _, agent = resume_from_checkpoint(
        checkpoint_path,
        iterate_value_q_pi,
        num_improvements=5,
        resume_args_mutator=resume_args_mutator
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_resume_from_checkpoint.pickle', 'wb') as file:
    #     pickle.dump(agent.pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_resume_from_checkpoint.pickle', 'rb') as file:
        pi_fixture = pickle.load(file)

    assert agent.pi == pi_fixture

    # rerun without resume and assert equal agents
    run_args = (
        '--random-seed 12345 --agent rlai.gpi.state_action_value.ActionValueMdpAgent '
        '--continuous-state-discretization-resolution 0.005 --gamma 0.95 '
        '--environment rlai.core.environments.gymnasium.Gym --gym-id CartPole-v1 --render-every-nth-episode 2 '
        '--train-function rlai.gpi.monte_carlo.iteration.iterate_value_q_pi --num-improvements 5 '
        '--num-episodes-per-improvement 2 --update-upon-every-visit True --epsilon 0.2 '
        '--q-S-A rlai.gpi.state_action_value.tabular.TabularStateActionValueEstimator '
        '--make-final-policy-greedy False --num-improvements-per-plot 2 --num-improvements-per-checkpoint 2 '
        f'--checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} '
        f'--save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'
    )
    _, full_agent_path = run(
        args=shlex.split(run_args),
        train_function_args_callback=train_function_args_callback
    )

    with open(full_agent_path, 'rb') as f:
        full_agent = pickle.load(f)

    assert full_agent.pi == agent.pi
