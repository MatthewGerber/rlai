import os
import pickle
import shlex
import tempfile
from typing import Dict

from numpy.random import RandomState

from rlai.core.environments.gymnasium import Gym
from rlai.gpi.monte_carlo.iteration import iterate_value_q_pi
from rlai.gpi.utils import resume_from_checkpoint
from rlai.runners.trainer import run
from test.rlai.utils import start_virtual_display_if_headless


def test_resume_gym_valid_environment():
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

    run_args = f'--random-seed 12345 --agent rlai.gpi.state_action_value.ActionValueMdpAgent --continuous-state-discretization-resolution 0.005 --gamma 0.95 --environment rlai.core.environments.gymnasium.Gym --gym-id CartPole-v1 --render-every-nth-episode 2 --train-function rlai.gpi.monte_carlo.iteration.iterate_value_q_pi --num-improvements 2 --num-episodes-per-improvement 2 --update-upon-every-visit True --epsilon 0.2 --q-S-A rlai.gpi.state_action_value.tabular.TabularStateActionValueEstimator --make-final-policy-greedy False --num-improvements-per-plot 2 --num-improvements-per-checkpoint 2 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'
    checkpoint_path, agent_path = run(
        args=shlex.split(run_args),
        train_function_args_callback=train_function_args_callback
    )

    random_state = RandomState(12345)
    resume_environment = Gym(random_state, None, 'CartPole-v1', None)
    agent = resume_from_checkpoint(
        checkpoint_path,
        iterate_value_q_pi,
        environment=resume_environment,
        num_improvements=2,
        resume_args_mutator=resume_args_mutator
    )

    resume_environment.close()

    # uncomment the following line and run test to update fixture
    with open(f'{os.path.dirname(__file__)}/fixtures/test_resume_gym_valid_environment.pickle', 'wb') as file:
        pickle.dump(agent.pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_resume_gym_valid_environment.pickle', 'rb') as file:
        pi_fixture = pickle.load(file)

    assert agent.pi == pi_fixture
