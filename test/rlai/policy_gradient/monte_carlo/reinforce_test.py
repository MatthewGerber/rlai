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


def test_resume():

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1.0 --environment rlai.environments.openai_gym.Gym --gym-id LunarLanderContinuous-v2 --plot-environment --T 500 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 2 --plot-state-value True --v-S rlai.v_S.function_approximation.estimators.ApproximateStateValueEstimator --feature-extractor rlai.environments.openai_gym.ContinuousLunarLanderFeatureExtractor --function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_loss --sgd-alpha 0.0 --learning-rate constant --eta0 0.0001 --policy rlai.policies.parameterized.continuous_action.ContinuousActionBetaDistributionPolicy --policy-feature-extractor rlai.environments.openai_gym.ContinuousLunarLanderFeatureExtractor --plot-policy --alpha 0.0001 --update-upon-every-visit True --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --num-episodes-per-checkpoint 1 --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name} --log DEBUG'))

    _, resumed_agent_path = run(shlex.split(f'--resume --random-seed 12345 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 2 --checkpoint-path {checkpoint_path} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    with open(resumed_agent_path, 'rb') as f:
        agent = pickle.load(f)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_resume.pickle', 'wb') as file:
    #     pickle.dump(agent, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_resume.pickle', 'rb') as file:
        agent_fixture = pickle.load(file)

    # assert that we get the expected result
    assert agent.pi == agent_fixture.pi
