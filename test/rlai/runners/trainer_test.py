import os
import pickle
import shlex
import tempfile
from typing import Any, Dict, Tuple

import pytest

import rlai.q_S_A.function_approximation.models
from rlai.policies.tabular import TabularPolicy
from rlai.runners.trainer import run
from test.rlai.utils import start_virtual_display_if_headless


def test_continuous_state_discretization():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ActionValueMdpAgent --continuous-state-discretization-resolution 0.1 --gamma 1 --environment rlai.environments.openai_gym.Gym --gym-id CartPole-v1 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --n-steps 10 --num-improvements 3 --num-episodes-per-improvement 5 --alpha 0.1 --epsilon 0.01 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 3 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_continuous_state_discretization.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_continuous_state_discretization.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_trajectory_sampling_planning():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ActionValueMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --planning-environment rlai.environments.mdp.TrajectorySamplingMdpPlanningEnvironment --num-planning-improvements-per-direct-improvement 10 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 5 --epsilon 0.01 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 10 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_trajectory_sampling_planning.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_trajectory_sampling_planning.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_prioritized_sweeping_planning_low_threshold():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ActionValueMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --planning-environment rlai.environments.mdp.PrioritizedSweepingMdpPlanningEnvironment --num-planning-improvements-per-direct-improvement 10 --priority-theta -1 --T-planning 50 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 5 --epsilon 0.01 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 10 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_prioritized_sweeping_planning_low_threshold.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_prioritized_sweeping_planning_low_threshold.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_prioritized_sweeping_planning_high_threshold():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ActionValueMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --planning-environment rlai.environments.mdp.PrioritizedSweepingMdpPlanningEnvironment --num-planning-improvements-per-direct-improvement 10 --priority-theta -10 --T-planning 50 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 1 --epsilon 0.01 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 10 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_prioritized_sweeping_planning_high_threshold.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_prioritized_sweeping_planning_high_threshold.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_q_learning_with_patsy_formula():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ActionValueMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --T 25 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 5 --num-episodes-per-improvement 5 --epsilon 0.05 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --verbose 1 --feature-extractor rlai.q_S_A.function_approximation.models.feature_extraction.StateActionIdentityFeatureExtractor --formula "C(s, levels={list(range(16))}):C(a, levels={list(range(4))})" --make-final-policy-greedy True --num-improvements-per-checkpoint 5 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_q_learning_with_patsy_formula.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_q_learning_with_patsy_formula.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_q_learning_with_state_action_interaction_feature_extractor():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ActionValueMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --T 25 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 5 --num-episodes-per-improvement 50 --epsilon 0.05 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --feature-extractor rlai.environments.gridworld.GridworldFeatureExtractor --make-final-policy-greedy True --num-improvements-per-checkpoint 5 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_q_learning_with_state_action_interaction_feature_extractor.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_q_learning_with_state_action_interaction_feature_extractor.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_sarsa_with_model_plots():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ActionValueMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --T 25 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode SARSA --num-improvements 10 --num-episodes-per-improvement 50 --epsilon 0.05 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --plot-model --plot-model-bins 10 --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --feature-extractor rlai.environments.gridworld.GridworldFeatureExtractor --make-final-policy-greedy True --num-improvements-per-checkpoint 5 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_sarsa_with_model_plots.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_sarsa_with_model_plots.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_continuous_action_discretization():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ActionValueMdpAgent --continuous-state-discretization-resolution 0.005 --gamma 0.95 --environment rlai.environments.openai_gym.Gym --gym-id MountainCarContinuous-v0 --T 20 --continuous-action-discretization-resolution 0.1 --render-every-nth-episode 2 --video-directory {tempfile.TemporaryDirectory().name} --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode SARSA --num-improvements 2 --num-episodes-per-improvement 1 --epsilon 0.01 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-plot 2 --num-improvements-per-checkpoint 2 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_continuous_action_discretization.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_continuous_action_discretization.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_gym_cartpole_function_approximation():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ActionValueMdpAgent --gamma 0.95 --environment rlai.environments.openai_gym.Gym --gym-id CartPole-v1 --render-every-nth-episode 2 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode SARSA --num-improvements 2 --num-episodes-per-improvement 2 --num-updates-per-improvement 1 --epsilon 0.2 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --loss squared_error --sgd-alpha 0.0 --learning-rate constant --eta0 0.001 --feature-extractor rlai.environments.openai_gym.CartpoleFeatureExtractor --make-final-policy-greedy True --num-improvements-per-plot 2 --num-improvements-per-checkpoint 2 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_gym_cartpole_function_approximation.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_gym_cartpole_function_approximation.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_gym_cartpole_tabular():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ActionValueMdpAgent --continuous-state-discretization-resolution 0.005 --gamma 0.95 --environment rlai.environments.openai_gym.Gym --gym-id CartPole-v1 --render-every-nth-episode 2 --train-function rlai.gpi.monte_carlo.iteration.iterate_value_q_pi --num-improvements 2 --num-episodes-per-improvement 2 --update-upon-every-visit True --epsilon 0.2 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-plot 2 --num-improvements-per-checkpoint 2 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_gym_cartpole_tabular.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_gym_cartpole_tabular.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_gym_cartpole_function_approximation_plot_model():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ActionValueMdpAgent --gamma 0.95 --environment rlai.environments.openai_gym.Gym --gym-id CartPole-v1 --render-every-nth-episode 2 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode SARSA --num-improvements 2 --num-episodes-per-improvement 2 --num-updates-per-improvement 1 --epsilon 0.2 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --plot-model --plot-model-bins 10 --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --loss squared_error --sgd-alpha 0.0 --learning-rate constant --eta0 0.001 --feature-extractor rlai.environments.openai_gym.CartpoleFeatureExtractor --make-final-policy-greedy True --num-improvements-per-plot 2 --num-improvements-per-checkpoint 2 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_gym_cartpole_function_approximation_plot_model.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_gym_cartpole_function_approximation_plot_model.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_gym_continuous_mountain_car():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ParameterizedMdpAgent --gamma 0.99 --environment rlai.environments.openai_gym.Gym --gym-id MountainCarContinuous-v0 --plot-environment --T 1000 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 2 --plot-state-value True --v-S rlai.v_S.function_approximation.estimators.ApproximateStateValueEstimator --feature-extractor rlai.environments.openai_gym.ContinuousMountainCarFeatureExtractor --function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_error --sgd-alpha 0.0 --learning-rate constant --eta0 0.01 --policy rlai.policies.parameterized.continuous_action.ContinuousActionBetaDistributionPolicy --policy-feature-extractor rlai.environments.openai_gym.ContinuousMountainCarFeatureExtractor --plot-policy --alpha 0.01 --update-upon-every-visit True --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --num-episodes-per-checkpoint 1 --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name} --log DEBUG'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_gym_continuous_mountain_car.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_gym_continuous_mountain_car.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_gridworld_plot_model_pdf():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ActionValueMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --T 25 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode SARSA --num-improvements 10 --num-episodes-per-improvement 50 --epsilon 0.05 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --plot-model --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --feature-extractor rlai.environments.gridworld.GridworldFeatureExtractor --make-final-policy-greedy True --num-improvements-per-checkpoint 5 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name} --pdf-save-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_gridworld_plot_model_pdf.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_gridworld_plot_model_pdf.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_scale_learning_rate_with_logging():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ActionValueMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --T 25 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 5 --num-episodes-per-improvement 50 --epsilon 0.05 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --scale-eta0-for-y --feature-extractor rlai.environments.gridworld.GridworldFeatureExtractor --make-final-policy-greedy True --num-improvements-per-checkpoint 5 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name} --log INFO'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_scale_learning_rate_with_logging.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_scale_learning_rate_with_logging.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_policy_gradient_reinforce_beta_with_baseline():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ParameterizedMdpAgent --gamma 0.99 --environment rlai.environments.openai_gym.Gym --gym-id LunarLanderContinuous-v2 --render-every-nth-episode 2 --plot-environment --T 2000 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 4 --v-S rlai.v_S.function_approximation.estimators.ApproximateStateValueEstimator --feature-extractor rlai.environments.openai_gym.ContinuousFeatureExtractor --function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_error --sgd-alpha 0.0 --learning-rate constant --eta0 0.00001 --policy rlai.policies.parameterized.continuous_action.ContinuousActionBetaDistributionPolicy --policy-feature-extractor rlai.environments.openai_gym.ContinuousFeatureExtractor --plot-policy --alpha 0.00001 --update-upon-every-visit True --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name} --log DEBUG'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_policy_gradient_reinforce_beta_with_baseline.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_policy_gradient_reinforce_beta_with_baseline.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_policy_gradient_reinforce_normal_with_baseline():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ParameterizedMdpAgent --gamma 0.99 --environment rlai.environments.openai_gym.Gym --gym-id LunarLanderContinuous-v2 --render-every-nth-episode 2 --steps-per-second 1000 --plot-environment --T 2000 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 4 --v-S rlai.v_S.function_approximation.estimators.ApproximateStateValueEstimator --feature-extractor rlai.environments.openai_gym.ContinuousFeatureExtractor --function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_error --sgd-alpha 0.0 --learning-rate constant --eta0 0.00001 --policy rlai.policies.parameterized.continuous_action.ContinuousActionNormalDistributionPolicy --policy-feature-extractor rlai.environments.openai_gym.ContinuousFeatureExtractor --plot-policy --alpha 0.00001 --update-upon-every-visit True --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_policy_gradient_reinforce_normal_with_baseline.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_policy_gradient_reinforce_normal_with_baseline.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_policy_gradient_reinforce_softmax_action_preferences_with_baseline():

    start_virtual_display_if_headless()

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ParameterizedMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 10 --v-S rlai.v_S.function_approximation.estimators.ApproximateStateValueEstimator --feature-extractor rlai.environments.gridworld.GridworldStateFeatureExtractor --function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_error --sgd-alpha 0.0 --learning-rate constant --eta0 0.001 --policy rlai.policies.parameterized.discrete_action.SoftMaxInActionPreferencesPolicy --policy-feature-extractor rlai.environments.gridworld.GridworldFeatureExtractor --alpha 0.001 --update-upon-every-visit False --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    _, agent = load_checkpoint_and_agent(checkpoint_path, agent_path)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_policy_gradient_reinforce_softmax_action_preferences_with_baseline.pickle', 'wb') as f:
    #     pickle.dump(agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_policy_gradient_reinforce_softmax_action_preferences_with_baseline.pickle', 'rb') as f:
        agent_fixture = pickle.load(f)

    assert_run(
        agent,
        agent_fixture
    )


def test_missing_arguments():

    run(shlex.split('--agent rlai.agents.mdp.ActionValueMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 5 --epsilon 0.01 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True'))


def test_unparsed_arguments():

    with pytest.raises(ValueError, match='Unparsed arguments'):
        run(shlex.split('--agent rlai.agents.mdp.ActionValueMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 5 --epsilon 0.01 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --XXXX'))


def test_help():

    with pytest.raises(ValueError, match='No training function specified. Cannot train.'):
        run(shlex.split('--agent rlai.agents.mdp.ActionValueMdpAgent --help'))


def test_resume():

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ActionValueMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 5 --epsilon 0.01 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 10 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    _, resumed_agent_path = run(shlex.split(f'--resume --random-seed 12345 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --num-improvements 10 --checkpoint-path {checkpoint_path} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    with open(resumed_agent_path, 'rb') as f:
        agent = pickle.load(f)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_resume.pickle', 'wb') as file:
    #     pickle.dump(agent, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_resume.pickle', 'rb') as file:
        agent_fixture = pickle.load(file)

    assert agent.pi == agent_fixture.pi


def test_too_many_coefficients_for_plot_model():

    old_vals = (
        rlai.q_S_A.function_approximation.models.MAX_PLOT_COEFFICIENTS,
        rlai.q_S_A.function_approximation.models.MAX_PLOT_ACTIONS
    )

    rlai.q_S_A.function_approximation.models.MAX_PLOT_COEFFICIENTS = 2
    rlai.q_S_A.function_approximation.models.MAX_PLOT_ACTIONS = 2

    run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.ActionValueMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --T 25 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode SARSA --num-improvements 10 --num-episodes-per-improvement 50 --epsilon 0.05 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --plot-model --plot-model-bins 10 --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --feature-extractor rlai.environments.gridworld.GridworldFeatureExtractor --make-final-policy-greedy True --num-improvements-per-checkpoint 5 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    (
        rlai.q_S_A.function_approximation.models.MAX_PLOT_COEFFICIENTS,
        rlai.q_S_A.function_approximation.models.MAX_PLOT_ACTIONS
    ) = old_vals


def assert_run(
        agent: Any,
        agent_fixture: Any
):
    """
    Assert test run.

    :param agent: Agent.
    :param agent_fixture: Agent fixture.
    """

    if isinstance(agent.pi, TabularPolicy):
        assert agent.q_S_A == agent_fixture.q_S_A
        assert agent.pi == agent_fixture.pi
    else:
        assert agent.pi == agent_fixture.pi


def load_checkpoint_and_agent(
        checkpoint_path: str,
        agent_path: str
) -> Tuple[Dict, Any]:
    """
    Load a checkpoint and agent from paths.

    :param checkpoint_path: Checkpoint path.
    :param agent_path: Agent path.
    :return: 2-tuple of checkpoint dictionary and agent object.
    """

    if checkpoint_path is None:
        checkpoint = None
    else:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

    with open(agent_path, 'rb') as f:
        agent = pickle.load(f)

    return checkpoint, agent
