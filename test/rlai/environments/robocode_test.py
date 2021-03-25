import json
import os
import pickle
import shlex
import socket
import tempfile
import time
from threading import Thread

import pytest

from rlai.environments.robocode import RobocodeFeatureExtractor
from rlai.runners.trainer import run


def test_learn():

    with open(f'{os.path.dirname(__file__)}/fixtures/test_robocode.pickle', 'rb') as file:
        state_sequence, fixture_pi, fixture_q_S_A = pickle.load(file)

    robocode_port = 54321

    # set up a mock robocode game that sends state sequence
    def robocode_mock():

        # wait for environment to start up and listen for connections
        time.sleep(5)

        t = 0
        while t < len(state_sequence):

            # start episode by connecting
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('127.0.0.1', robocode_port))

                try:

                    while t < len(state_sequence):

                        # send the current game state in the sequence
                        state_dict_json = state_sequence[t]
                        s.sendall(state_dict_json.encode('utf-8'))
                        t += 1

                        # receive next action
                        s.recv(99999999)

                        # if the next state starts a new episode, then break.
                        if t < len(state_sequence):
                            next_state_dict = json.loads(state_sequence[t])
                            if next_state_dict['state']['time'] == 0:
                                break

                # if environment closes connection during receive, it ends the episode.
                except Exception:
                    pass

    robocode_mock_thread = Thread(target=robocode_mock)
    robocode_mock_thread.start()

    # run training and load resulting agent
    agent_path = tempfile.NamedTemporaryFile(delete=False).name
    cmd = f'--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 0.9 --environment rlai.environments.robocode.RobocodeEnvironment --port {robocode_port} --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode SARSA --n-steps 100 --num-improvements 2 --num-episodes-per-improvement 1 --num-updates-per-improvement 1 --epsilon 0.15 --q-S-A rlai.value_estimation.function_approximation.estimators.ApproximateStateActionValueEstimator --plot-model --plot-model-bins 10 --plot-model-per-improvements 10 --function-approximation-model rlai.value_estimation.function_approximation.models.sklearn.SKLearnSGD --feature-extractor rlai.environments.robocode.RobocodeFeatureExtractor --make-final-policy-greedy True --num-improvements-per-plot 5 --save-agent-path {agent_path}'
    run(shlex.split(cmd))
    robocode_mock_thread.join()
    with open(agent_path, 'rb') as f:
        agent = pickle.load(f)

    # uncomment the following line and run test to update fixture. you'll need to also uncomment the bit of code in
    # TcpMdpEnvironment.read_from_client that stores the state sequence.
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_robocode.pickle', 'wb') as file:
    #     pickle.dump((robocode.state_sequence, mdp_agent.pi, q_S_A), file)

    assert agent.pi == fixture_pi


def test_feature_extractor():

    assert RobocodeFeatureExtractor.is_clockwise_move(0, 1)
    assert RobocodeFeatureExtractor.is_clockwise_move(-1, 1)
    assert not RobocodeFeatureExtractor.is_clockwise_move(1, 1)
    assert RobocodeFeatureExtractor.get_shortest_degree_change(1, -1) == -2
    assert RobocodeFeatureExtractor.get_shortest_degree_change(-1, 1) == 2
    assert RobocodeFeatureExtractor.get_shortest_degree_change(1, 1) == 0
    assert RobocodeFeatureExtractor.normalize(366) == 6
    assert RobocodeFeatureExtractor.normalize(-5) == 355

    with pytest.raises(ValueError, match='Failed to normalize degrees'):
        assert RobocodeFeatureExtractor.normalize(1000)
