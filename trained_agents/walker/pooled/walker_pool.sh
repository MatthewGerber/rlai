#!/bin/bash

rm -f kill_walker_pool.sh
rm -f walker_pool.log

num_runners=10
eta_values=(0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001)
alpha_values=(0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001)

for n in $(seq 0 $((num_runners - 1))); do
    rlai train --random-seed "1234${n}" --agent rlai.agents.mdp.ParameterizedMdpAgent --gamma 1.0 --environment rlai.environments.openai_gym.Gym --gym-id BipedalWalker-v3 --render-every-nth-episode 100 --video-directory "$HOME/Desktop/walker_videos_${n}" --force --T 500 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 4000 --v-S rlai.v_S.function_approximation.estimators.ApproximateStateValueEstimator --feature-extractor rlai.environments.openai_gym.ContinuousFeatureExtractor --function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_loss --sgd-alpha 0.0 --learning-rate constant --eta0 "${eta_values[$n]}" --policy rlai.policies.parameterized.continuous_action.ContinuousActionBetaDistributionPolicy --policy-feature-extractor rlai.environments.openai_gym.ContinuousFeatureExtractor --alpha "${alpha_values[$n]}" --update-upon-every-visit True --save-agent-path "$HOME/Desktop/walker_agent_${n}.pickle" --log INFO --training-pool-directory ~/Desktop/walker_pool --training-pool-count ${num_runners} --training-pool-iterate-episodes 5 --training-pool-evaluate-episodes 5 >>walker_pool.log 2>&1 &
    echo "sudo kill -9 $!" >> kill_walker_pool.sh
done