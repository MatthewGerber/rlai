import os
import pickle
import shlex
import tempfile

import numpy as np

from rlai.runners.trainer import run


def test_manual_versus_jax_policy_gradient():
    """
    Test.
    """

    manual_agent_path = tempfile.NamedTemporaryFile(delete=False).name
    run(shlex.split(f'--random-seed 12345 --agent rlai.policy_gradient.ParameterizedMdpAgent --gamma 1 --environment rlai.core.environments.gridworld.Gridworld --id example_4_1 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 10 --policy rlai.policy_gradient.policies.discrete_action.SoftMaxInActionPreferencesPolicy --policy-feature-extractor rlai.core.environments.gridworld.GridworldFeatureExtractor --alpha 0.0001 --update-upon-every-visit True --num-episodes-per-policy-update-plot 1 --policy-update-plot-pdf-directory {tempfile.NamedTemporaryFile(delete=True).name} --save-agent-path {manual_agent_path} --log DEBUG'))
    with open(manual_agent_path, 'rb') as f:
        manual_agent = pickle.load(f)

    jax_agent_path = tempfile.NamedTemporaryFile(delete=False).name
    run(shlex.split(f'--random-seed 12345 --agent rlai.policy_gradient.ParameterizedMdpAgent --gamma 1 --environment rlai.core.environments.gridworld.Gridworld --id example_4_1 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 10 --policy rlai.policy_gradient.policies.discrete_action.SoftMaxInActionPreferencesJaxPolicy --policy-feature-extractor rlai.core.environments.gridworld.GridworldFeatureExtractor --alpha 0.0001 --update-upon-every-visit True --save-agent-path {jax_agent_path} --log DEBUG'))
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
    """
    Test.
    """

    checkpoint_path, agent_path = run(shlex.split(
        '--random-seed 12345 --agent rlai.policy_gradient.ParameterizedMdpAgent --gamma 1.0 '
        '--environment rlai.core.environments.gymnasium.Gym --gym-id LunarLanderContinuous-v3 --T 500 '
        '--train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 2 '
        '--v-S rlai.state_value.function_approximation.ApproximateStateValueEstimator '
        '--feature-extractor rlai.core.environments.gymnasium.ContinuousLunarLanderFeatureExtractor '
        '--function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_error --sgd-alpha 0.0 '
        '--learning-rate constant --eta0 0.0001 '
        '--policy rlai.policy_gradient.policies.continuous_action.ContinuousActionBetaDistributionPolicy '
        '--policy-feature-extractor rlai.core.environments.gymnasium.ContinuousLunarLanderFeatureExtractor '
        '--alpha 0.0001 --update-upon-every-visit True '
        f'--checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --num-episodes-per-checkpoint 1 '
        f'--save-agent-path {tempfile.NamedTemporaryFile(delete=False).name} --log DEBUG'
    ))

    checkpoint_path, _ = run(shlex.split(
        '--resume --train-function rlai.policy_gradient.monte_carlo.reinforce.improve '
        f'--num-episodes 5 --checkpoint-path {checkpoint_path} --num-episodes-per-checkpoint 1 '
        f'--save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'
    ))

    _, resumed_agent_path = run(shlex.split(
        '--resume --train-function rlai.policy_gradient.monte_carlo.reinforce.improve '
        f'--num-episodes 10 --start-episode 9 --checkpoint-path {checkpoint_path} '
        f'--save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'
    ))

    with open(resumed_agent_path, 'rb') as f:
        agent = pickle.load(f)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_resume.pickle', 'wb') as file:
    #     pickle.dump(agent, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_resume.pickle', 'rb') as file:
        agent_fixture = pickle.load(file)

    # assert that we get the expected result
    assert agent.pi == agent_fixture.pi

    # run the full number of episodes and check equal agents
    _, full_agent_path = run(shlex.split(
        '--random-seed 12345 --agent rlai.policy_gradient.ParameterizedMdpAgent --gamma 1.0 '
        '--environment rlai.core.environments.gymnasium.Gym --gym-id LunarLanderContinuous-v3 --T 500 '
        '--train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 7 '
        '--v-S rlai.state_value.function_approximation.ApproximateStateValueEstimator '
        '--feature-extractor rlai.core.environments.gymnasium.ContinuousLunarLanderFeatureExtractor '
        '--function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_error --sgd-alpha 0.0 '
        '--learning-rate constant --eta0 0.0001 '
        '--policy rlai.policy_gradient.policies.continuous_action.ContinuousActionBetaDistributionPolicy '
        '--policy-feature-extractor rlai.core.environments.gymnasium.ContinuousLunarLanderFeatureExtractor '
        '--alpha 0.0001 --update-upon-every-visit True '
        f'--checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --num-episodes-per-checkpoint 1 '
        f'--save-agent-path {tempfile.NamedTemporaryFile(delete=False).name} --log DEBUG'
    ))

    with open(full_agent_path, 'rb') as f:
        full_agent = pickle.load(f)

    assert full_agent.pi == agent.pi
