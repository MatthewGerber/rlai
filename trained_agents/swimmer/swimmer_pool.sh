#!/bin/bash

for n in $(seq 1 10); do
    rlai train --random-seed 1234${n} --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1.0 --environment rlai.environments.openai_gym.Gym --gym-id Swimmer-v2 --render-every-nth-episode 500 --video-directory ~/Desktop/swimmer_videos_${n} --force --T 500 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 50000 --v-S rlai.v_S.function_approximation.estimators.ApproximateStateValueEstimator --feature-extractor rlai.environments.openai_gym.SignedCodingFeatureExtractor --function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_loss --sgd-alpha 0.0 --learning-rate constant --eta0 0.001 --policy rlai.policies.parameterized.continuous_action.ContinuousActionBetaDistributionPolicy --policy-feature-extractor rlai.environments.openai_gym.SignedCodingFeatureExtractor --alpha 0.0001 --update-upon-every-visit True --num-episodes-per-checkpoint 500 --checkpoint-path ~/Desktop/swimmer_${n}_checkpoint.pickle --save-agent-path ~/Desktop/swimmer_agent_${n}.pickle --log INFO --training-pool-directory ~/Desktop/swimmer_pool --training-pool-slot ${n},10 &
done