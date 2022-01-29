# MuJoCo Swimming Worm
* Content
{:toc}

# Introduction

You can read more about this environment [here](https://gym.openai.com/envs/Swimmer-v2/). Many of the 
issues involved in solving this environment are addressed in the 
[continuous mountain car](mountain_car_continuous.md) and 
[continuous lunar lander](lunar_lander_continuous.md) case studies, so we will focus here on details specific to the
MuJoCo swimming worm environment. There are not many. Except for a minor change of learning rates, this problem was 
solved with a direct application of the same policy gradient method detailed in the prior case studies. The training 
time was significantly greater, but the environment itself is simpler:  no fuel levels and a simpler reward function.

# Development
A few key points of development are worth mentioning.

### Feature Space
As described in 
[the code](https://github.com/openai/gym/blob/39ba73ff17ce4df2ea08c1f9ab88f0726a476afb/gym/envs/mujoco/swimmer.py#L70-L79)
for this environment, the worm is characterized but the expected set of angles and velocities, all of which are in the 
range `[-infinity, +infinity]`. As usual in this sort of state space, the right action to take at a given time likely 
depends on all state variables that time. When all variables are continuous as here, a straightforward way of encoding 
this dependence is to form a discrete category for each possible combination of state-variable signs. With an 
8-dimensional state space, this results in `2^8=256` state categories. Only one of these categories is active at any 
particular time.

### Baseline and Policy Gradient Learning Rates

# Training
The following command trains an agent for the MuJoCo swimming worm environment using policy gradient optimization 
with a baseline state-value estimator:

```
rlai train --random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1.0 --environment rlai.environments.openai_gym.Gym --gym-id Swimmer-v2 --render-every-nth-episode 500 --video-directory ~/Desktop/swimmer_videos --force --T 500 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 50000 --v-S rlai.v_S.function_approximation.estimators.ApproximateStateValueEstimator --feature-extractor rlai.environments.openai_gym.SignedCodingFeatureExtractor --function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_loss --sgd-alpha 0.0 --learning-rate constant --eta0 0.001 --policy rlai.policies.parameterized.continuous_action.ContinuousActionBetaDistributionPolicy --policy-feature-extractor rlai.environments.openai_gym.SignedCodingFeatureExtractor --alpha 0.00001 --update-upon-every-visit True --num-episodes-per-checkpoint 500 --checkpoint-path ~/Desktop/swimmer_checkpoint.pickle --save-agent-path ~/Desktop/swimmer_agent.pickle --log DEBUG
```

The argument are explained below.

### RLAI
* `train`:  Train the agent. 
* `--random-seed 12345`:  For reproducibility.

### Agent
* `--agent rlai.agents.mdp.StochasticMdpAgent`:  Standard stochastic MDP agent. 
* `--gamma 1.0`:  Do not discount.

### Environment
* `--environment rlai.environments.openai_gym.Gym`:  Environment class.
* `--gym-id Swimmer-v2`:  OpenAI Gym environment identifier.
* `--render-every-nth-episode 500`:  Render a video every 500 episodes.
* `--video-directory ~/Desktop/swimmer_videos`:  Where to store rendered videos.
* `--force`:  Overwrite videos in the video directory.
* `--T 500`:  Limit episodes to 500 steps.

### Training Function and Episodes
* `--train-function rlai.policy_gradient.monte_carlo.reinforce.improve`:  Run the REINFORCE policy gradient optimization
algorithm.
* `--num-episodes 50000`:  Run 50000 episodes.

### Baseline State-Value Estimator
* `--v-S rlai.v_S.function_approximation.estimators.ApproximateStateValueEstimator`:  Baseline state-value estimator.  
* `--feature-extractor rlai.environments.openai_gym.SignedCodingFeatureExtractor`:  Feature extractor for the
baseline state-value estimator.
* `--function-approximation-model rlai.models.sklearn.SKLearnSGD`:  Use SKLearn's SGD for the baseline state-value 
estimator.
* `--loss squared_loss`:  Use a squared loss within the baseline state-value estimator.
* `--sgd-alpha 0.0`:  Do not use regularization.
* `--learning-rate constant`:  Use a constant learning rate schedule.
* `--eta0 0.001`:  Learning rate.

### Policy
* `--policy rlai.policies.parameterized.continuous_action.ContinuousActionBetaDistributionPolicy`:  Use the beta
distribution to model the action-density distribution within the policy.
* `--policy-feature-extractor rlai.environments.openai_gym.SignedCodingFeatureExtractor`:  Feature extractor
for the policy gradient optimizer.
* `--alpha 0.00001`:  Learning rate for policy gradient updates.
* `--update-upon-every-visit True`:  Update the policy's action-density distribution every time a state is encountered
  (as opposed to the first visit only).

### Output
* `--num-episodes-per-checkpoint 500`:  Store a resumable checkpoint file every 500 episodes.
* `--checkpoint-path ~/Desktop/swimmer_checkpoint.pickle`:  Where to store checkpoint files (an index is inserted).
* `--save-agent-path ~/Desktop/swimmer_agent.pickle`:  Where to save the final agent.
* `~/Desktop/swimmer_agent.pickle --log DEBUG`:  Produce debug logging output.

# Results

The following sequence of videos shows the progression of swimmer policies:

### First Agent:  Random
{% include youtubePlayer.html id="ZeUbC9U9Zas" %}

### First Agent:  Scooping Segment
{% include youtubePlayer.html id="pnHlOhsRq6A" %}

### First Agent:  Improved Front Oscillations
{% include youtubePlayer.html id="sNRsMgv8FGI" %}

### First Agent:  Coordinated Segments
{% include youtubePlayer.html id="tRwPN34U1fk" %}

### First Agent:  Failure
{% include youtubePlayer.html id="fjqUQm8iKXA" %}

### First Agent:  Final Agent
{% include youtubePlayer.html id="9a0jYLADr-c" %}
