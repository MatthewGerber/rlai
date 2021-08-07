import os
import pickle
import shlex
import tempfile

import numpy as np

from rlai.runners.trainer import run


def test_manual_versus_jax_policy_gradient():

    manual_agent_path = tempfile.NamedTemporaryFile(delete=False).name
    run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 10 --policy rlai.policies.parameterized.discrete_action.SoftMaxInActionPreferencesPolicy --policy-feature-extractor rlai.environments.gridworld.GridworldFeatureExtractor --alpha 0.0001 --update-upon-every-visit True --save-agent-path {manual_agent_path} --log DEBUG'))
    with open(manual_agent_path, 'rb') as f:
        manual_agent = pickle.load(f)

    jax_agent_path = tempfile.NamedTemporaryFile(delete=False).name
    run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 10 --policy rlai.policies.parameterized.discrete_action.SoftMaxInActionPreferencesJaxPolicy --policy-feature-extractor rlai.environments.gridworld.GridworldFeatureExtractor --alpha 0.0001 --update-upon-every-visit True --save-agent-path {jax_agent_path} --log DEBUG'))
    with open(jax_agent_path, 'rb') as f:
        jax_agent = pickle.load(f)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_manual_versus_jax_policy_gradient.pickle', 'wb') as file:
    #     pickle.dump(jax_agent, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_manual_versus_jax_policy_gradient.pickle', 'rb') as file:
        fixture_agent = pickle.load(file)

    assert np.allclose(manual_agent.pi.theta, jax_agent.pi.theta)
    assert np.allclose(jax_agent.pi.theta, fixture_agent.pi.theta)
