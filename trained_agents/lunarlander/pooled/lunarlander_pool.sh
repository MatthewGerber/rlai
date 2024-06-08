#!/bin/bash

rm -f kill_lunarlander_pool.sh
rm -f lunarlander_pool.log

num_runners=10
eta_values=(0.00001 0.0001 0.0001 0.0001 0.001 0.001 0.001 0.001 0.01 0.01)
alpha_values=(0.00001 0.0001 0.0001 0.0001 0.001 0.001 0.001 0.001 0.01 0.01)

for n in $(seq 0 $((num_runners - 1))); do
    rlai train --random-seed "1234${n}" --agent rlai.policy_gradient.ParameterizedMdpAgent --gamma 1.0 --environment rlai.core.environments.gymnasium.Gym --gym-id LunarLanderContinuous-v3 --render-every-nth-episode 1000 --video-directory "$HOME/Desktop/lunarlander_videos_${n}" --T 500 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 50000 --v-S rlai.state_value.function_approximation.ApproximateStateValueEstimator --feature-extractor rlai.core.environments.gymnasium.ContinuousLunarLanderFeatureExtractor --function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_error --sgd-alpha 0.001 --learning-rate constant --eta0 "${eta_values[$n]}" --policy rlai.policy_gradient.policies.continuous_action.ContinuousActionBetaDistributionPolicy --policy-feature-extractor rlai.core.environments.gymnasium.ContinuousLunarLanderFeatureExtractor --alpha "${alpha_values[$n]}" --update-upon-every-visit True --save-agent-path "$HOME/Desktop/lunarlander_agent_${n}.pickle" --log INFO --training-pool-directory ~/Desktop/lunarlander_pool --training-pool-count ${num_runners} --training-pool-iterate-episodes 100 --training-pool-evaluate-episodes 100 --training-pool-max-iterations-without-improvement 50 >>lunarlander_pool.log 2>&1 &
    echo "sudo kill -9 $!" >> kill_lunarlander_pool.sh
done