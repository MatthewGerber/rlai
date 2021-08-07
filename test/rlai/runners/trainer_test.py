import os
import pickle
import shlex
import tempfile

import pytest

import rlai.q_S_A.function_approximation.models
from rlai.policies.tabular import TabularPolicy
from rlai.runners.trainer import run
from test.rlai.utils import init_virtual_display


def test_run():

    display = init_virtual_display()

    run_args_list = [
        f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --continuous-state-discretization-resolution 0.1 --gamma 1 --environment rlai.environments.openai_gym.Gym --gym-id CartPole-v1 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --n-steps 10 --num-improvements 3 --num-episodes-per-improvement 5 --alpha 0.1 --epsilon 0.01 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 3 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --planning-environment rlai.environments.mdp.TrajectorySamplingMdpPlanningEnvironment --num-planning-improvements-per-direct-improvement 10 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 5 --epsilon 0.01 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 10 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --planning-environment rlai.environments.mdp.PrioritizedSweepingMdpPlanningEnvironment --num-planning-improvements-per-direct-improvement 10 --priority-theta -1 --T-planning 50 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 5 --epsilon 0.01 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 10 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --planning-environment rlai.environments.mdp.PrioritizedSweepingMdpPlanningEnvironment --num-planning-improvements-per-direct-improvement 10 --priority-theta -10 --T-planning 50 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 1 --epsilon 0.01 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 10 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --T 25 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 5 --num-episodes-per-improvement 5 --epsilon 0.05 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --verbose 1 --feature-extractor rlai.q_S_A.function_approximation.models.feature_extraction.StateActionIdentityFeatureExtractor --formula "C(s, levels={list(range(16))}):C(a, levels={list(range(4))})" --make-final-policy-greedy True --num-improvements-per-checkpoint 5 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --T 25 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 5 --num-episodes-per-improvement 50 --epsilon 0.05 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --feature-extractor rlai.environments.gridworld.GridworldFeatureExtractor --make-final-policy-greedy True --num-improvements-per-checkpoint 5 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --T 25 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode SARSA --num-improvements 10 --num-episodes-per-improvement 50 --epsilon 0.05 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --plot-model --plot-model-bins 10 --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --feature-extractor rlai.environments.gridworld.GridworldFeatureExtractor --make-final-policy-greedy True --num-improvements-per-checkpoint 5 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --continuous-state-discretization-resolution 0.005 --gamma 0.95 --environment rlai.environments.openai_gym.Gym --gym-id MountainCarContinuous-v0 --T 20 --continuous-action-discretization-resolution 0.1 --render-every-nth-episode 2 --video-directory {tempfile.TemporaryDirectory().name} --force --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode SARSA --num-improvements 2 --num-episodes-per-improvement 1 --epsilon 0.01 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-plot 2 --num-improvements-per-checkpoint 2 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 0.95 --environment rlai.environments.openai_gym.Gym --gym-id CartPole-v1 --render-every-nth-episode 2 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode SARSA --num-improvements 2 --num-episodes-per-improvement 2 --num-updates-per-improvement 1 --epsilon 0.2 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --loss squared_loss --sgd-alpha 0.0 --learning-rate constant --eta0 0.001 --feature-extractor rlai.environments.openai_gym.CartpoleFeatureExtractor --make-final-policy-greedy True --num-improvements-per-plot 2 --num-improvements-per-checkpoint 2 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --continuous-state-discretization-resolution 0.005 --gamma 0.95 --environment rlai.environments.openai_gym.Gym --gym-id CartPole-v1 --render-every-nth-episode 2 --train-function rlai.gpi.monte_carlo.iteration.iterate_value_q_pi --num-improvements 2 --num-episodes-per-improvement 2 --update-upon-every-visit True --epsilon 0.2 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-plot 2 --num-improvements-per-checkpoint 2 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 0.95 --environment rlai.environments.openai_gym.Gym --gym-id CartPole-v1 --render-every-nth-episode 2 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode SARSA --num-improvements 2 --num-episodes-per-improvement 2 --num-updates-per-improvement 1 --epsilon 0.2 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --plot-model --plot-model-bins 10 --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --loss squared_loss --sgd-alpha 0.0 --learning-rate constant --eta0 0.001 --feature-extractor rlai.environments.openai_gym.CartpoleFeatureExtractor --make-final-policy-greedy True --num-improvements-per-plot 2 --num-improvements-per-checkpoint 2 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --T 25 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode SARSA --num-improvements 10 --num-episodes-per-improvement 50 --epsilon 0.05 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --plot-model --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --feature-extractor rlai.environments.gridworld.GridworldFeatureExtractor --make-final-policy-greedy True --num-improvements-per-checkpoint 5 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name} --pdf-save-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --T 25 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 5 --num-episodes-per-improvement 50 --epsilon 0.05 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --scale-eta0-for-y --feature-extractor rlai.environments.gridworld.GridworldFeatureExtractor --make-final-policy-greedy True --num-improvements-per-checkpoint 5 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name} --log INFO',
        f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 0.99 --environment rlai.environments.openai_gym.Gym --gym-id LunarLanderContinuous-v2 --render-every-nth-episode 2 --plot-environment --T 2000 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 4 --v-S rlai.v_S.function_approximation.estimators.ApproximateStateValueEstimator --feature-extractor rlai.environments.openai_gym.ContinuousFeatureExtractor --function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_loss --sgd-alpha 0.0 --learning-rate constant --eta0 0.00001 --policy rlai.policies.parameterized.continuous_action.ContinuousActionBetaDistributionPolicy --policy-feature-extractor rlai.environments.openai_gym.ContinuousFeatureExtractor --plot-policy --alpha 0.00001 --update-upon-every-visit True --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name} --log DEBUG',
        f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 0.99 --environment rlai.environments.openai_gym.Gym --gym-id LunarLanderContinuous-v2 --render-every-nth-episode 2 --steps-per-second 1000 --plot-environment --T 2000 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 4 --v-S rlai.v_S.function_approximation.estimators.ApproximateStateValueEstimator --feature-extractor rlai.environments.openai_gym.ContinuousFeatureExtractor --function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_loss --sgd-alpha 0.0 --learning-rate constant --eta0 0.00001 --policy rlai.policies.parameterized.continuous_action.ContinuousActionNormalDistributionPolicy --policy-feature-extractor rlai.environments.openai_gym.ContinuousFeatureExtractor --plot-policy --alpha 0.00001 --update-upon-every-visit True --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 10 --v-S rlai.v_S.function_approximation.estimators.ApproximateStateValueEstimator --feature-extractor rlai.environments.gridworld.GridworldStateFeatureExtractor --function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_loss --sgd-alpha 0.0 --learning-rate constant --eta0 0.001 --policy rlai.policies.parameterized.discrete_action.SoftMaxInActionPreferencesPolicy --policy-feature-extractor rlai.environments.gridworld.GridworldFeatureExtractor --alpha 0.001 --update-upon-every-visit False --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'
    ]

    run_checkpoint_agent = {}

    for run_args in run_args_list:

        checkpoint_path, agent_path = run(shlex.split(run_args))

        if checkpoint_path is None:
            checkpoint = None
        else:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)

        with open(agent_path, 'rb') as f:
            agent = pickle.load(f)

        run_checkpoint_agent[run_args] = checkpoint, agent

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/trainer_test.pickle', 'wb') as f:
    #     pickle.dump(run_checkpoint_agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/trainer_test.pickle', 'rb') as f:
        run_fixture = pickle.load(f)

    passed_all = True
    for run_args, run_args_fixture in zip(run_args_list, run_fixture.keys()):

        print(f'Checking test results for run {run_args}...', end='')

        checkpoint, agent = run_checkpoint_agent[run_args]
        checkpoint_fixture, agent_fixture = run_fixture[run_args_fixture]

        try:

            if isinstance(agent.pi, TabularPolicy):
                assert checkpoint['q_S_A'] == checkpoint_fixture['q_S_A']
                assert agent.pi == agent_fixture.pi
            else:
                assert agent.pi == agent_fixture.pi

            print('passed.')

        except Exception:
            passed_all = False
            print(f'failed')

    assert passed_all
    assert len(run_args_list) == len(run_fixture.keys())


def test_missing_arguments():

    run(shlex.split('--agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 5 --epsilon 0.01 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True'))


def test_unparsed_arguments():

    with pytest.raises(ValueError, match='Unparsed arguments'):
        run(shlex.split('--agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 5 --epsilon 0.01 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --XXXX'))


def test_help():

    with pytest.raises(ValueError, match='No training function specified. Cannot train.'):
        run(shlex.split('--agent rlai.agents.mdp.StochasticMdpAgent --help'))


def test_resume():

    checkpoint_path, agent_path = run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 5 --epsilon 0.01 --q-S-A rlai.q_S_A.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 10 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

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

    run(shlex.split(f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --T 25 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode SARSA --num-improvements 10 --num-episodes-per-improvement 50 --epsilon 0.05 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --plot-model --plot-model-bins 10 --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --feature-extractor rlai.environments.gridworld.GridworldFeatureExtractor --make-final-policy-greedy True --num-improvements-per-checkpoint 5 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'))

    (
        rlai.q_S_A.function_approximation.models.MAX_PLOT_COEFFICIENTS,
        rlai.q_S_A.function_approximation.models.MAX_PLOT_ACTIONS
    ) = old_vals
