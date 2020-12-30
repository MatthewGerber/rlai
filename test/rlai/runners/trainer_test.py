import os
import pickle
import shlex
import tempfile

import numpy as np

from rlai.policies.tabular import TabularPolicy
from rlai.runners.trainer import run


def test_run():

    run_args_list = [
        f'--agent rlai.agents.mdp.StochasticMdpAgent --continuous-state-discretization-resolution 0.1 --gamma 1 --environment rlai.environments.openai_gym.Gym --gym-id CartPole-v1 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --n-steps 10 --num-improvements 3 --num-episodes-per-improvement 5 --alpha 0.1 --epsilon 0.01 --q-S-A rlai.value_estimation.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 3 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --planning-environment rlai.environments.mdp.TrajectorySamplingMdpPlanningEnvironment --num-planning-improvements-per-direct-improvement 10 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 5 --epsilon 0.01 --q-S-A rlai.value_estimation.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 10 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --planning-environment rlai.environments.mdp.PrioritizedSweepingMdpPlanningEnvironment --num-planning-improvements-per-direct-improvement 10 --priority-theta 0.1 --T-planning 50 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 5 --epsilon 0.01 --q-S-A rlai.value_estimation.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 10 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --n-steps 1 --num-improvements 5 --num-episodes-per-improvement 5 --epsilon 0.05 --q-S-A rlai.value_estimation.function_approximation.estimators.ApproximateStateActionValueEstimator --function-approximation-model rlai.value_estimation.function_approximation.models.sklearn.SKLearnSGD --feature-extractor rlai.value_estimation.function_approximation.models.feature_extraction.StateActionIdentityFeatureExtractor --formula "C(s, levels={list(range(16))}):C(a, levels={list(range(4))})" --make-final-policy-greedy True --num-improvements-per-checkpoint 5 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --n-steps 5 --num-improvements 5 --num-episodes-per-improvement 50 --epsilon 0.05 --q-S-A rlai.value_estimation.function_approximation.estimators.ApproximateStateActionValueEstimator --function-approximation-model rlai.value_estimation.function_approximation.models.sklearn.SKLearnSGD --feature-extractor rlai.environments.gridworld.GridworldFeatureExtractor --make-final-policy-greedy True --num-improvements-per-checkpoint 5 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'
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

    assert len(run_args_list) == len(run_fixture.keys())

    for run_args, run_args_fixture in zip(run_args_list, run_fixture.keys()):

        print(f'Checking test results for run {run_args}...', end='')

        checkpoint, agent = run_checkpoint_agent[run_args]
        checkpoint_fixture, agent_fixture = run_fixture[run_args_fixture]

        if isinstance(agent.pi, TabularPolicy):
            assert checkpoint['q_S_A'] == checkpoint_fixture['q_S_A']
            assert agent.pi == agent_fixture.pi
        else:
            assert np.allclose(agent.pi.estimator.model.model.coef_, agent_fixture.pi.estimator.model.model.coef_)

        print('passed.')
