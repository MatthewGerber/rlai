import os
import pickle
import shlex
import tempfile

import pytest
from numpy.random import RandomState

from rlai.environments.openai_gym import Gym
from rlai.gpi.monte_carlo.iteration import iterate_value_q_pi
from rlai.gpi.utils import resume_from_checkpoint
from rlai.runners.trainer import run
from test.rlai.utils import init_virtual_display


def test_resume_gym_invalid_environment():

    init_virtual_display()

    run_args = f'--agent rlai.agents.mdp.StochasticMdpAgent --continuous-state-discretization-resolution 0.005 --gamma 0.95 --environment rlai.environments.openai_gym.Gym --gym-id CartPole-v1 --render-every-nth-episode 2 --train-function rlai.gpi.monte_carlo.iteration.iterate_value_q_pi --num-improvements 2 --num-episodes-per-improvement 2 --update-upon-every-visit True --epsilon 0.2 --q-S-A rlai.value_estimation.tabular.TabularStateActionValueEstimator --make-final-policy-greedy False --num-improvements-per-plot 2 --num-improvements-per-checkpoint 2 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'

    checkpoint_path, agent_path = run(shlex.split(run_args))

    with pytest.raises(ValueError, match='No environment passed when resuming an assumed OpenAI Gym environment.'):
        resume_from_checkpoint(
            checkpoint_path,
            iterate_value_q_pi,
        )

    resume_environment = Gym(RandomState(), None, 'Acrobot-v1', None)
    with pytest.raises(ValueError, match='Attempted to resume'):
        resume_from_checkpoint(
            checkpoint_path,
            iterate_value_q_pi,
            environment=resume_environment
        )


def test_resume_gym_valid_environment():

    init_virtual_display()

    run_args = f'--agent rlai.agents.mdp.StochasticMdpAgent --continuous-state-discretization-resolution 0.005 --gamma 0.95 --environment rlai.environments.openai_gym.Gym --gym-id CartPole-v1 --render-every-nth-episode 2 --train-function rlai.gpi.monte_carlo.iteration.iterate_value_q_pi --num-improvements 2 --num-episodes-per-improvement 2 --update-upon-every-visit True --epsilon 0.2 --q-S-A rlai.value_estimation.tabular.TabularStateActionValueEstimator --make-final-policy-greedy False --num-improvements-per-plot 2 --num-improvements-per-checkpoint 2 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'
    checkpoint_path, agent_path = run(shlex.split(run_args))
    random_state = RandomState(12345)
    resume_environment = Gym(random_state, None, 'CartPole-v1', None)
    agent = resume_from_checkpoint(
        checkpoint_path,
        iterate_value_q_pi,
        environment=resume_environment,
        num_improvements=2
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_resume_gym_valid_environment.pickle', 'wb') as file:
    #     pickle.dump(agent.pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_resume_gym_valid_environment.pickle', 'rb') as file:
        pi_fixture = pickle.load(file)

    assert agent.pi == pi_fixture
