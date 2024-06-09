[Home](../index.md) > MuJoCo Swimming Worm
* Content
{:toc}

# Introduction
You can read more about this environment [here](https://gymnasium.farama.org/environments/mujoco/swimmer/). Many of the 
issues involved in solving this environment are addressed in the 
[continuous mountain car](mountain_car_continuous.md) and 
[continuous lunar lander](lunar_lander_continuous.md) case studies, so we will focus here on details specific to the
MuJoCo swimming worm environment. There are not many. Except for a minor change of learning rates, this problem was 
solved with a direct application of the same policy gradient method detailed in the prior case studies. The training 
time was significantly greater, but the environment itself is simpler:  no fuel levels and a simpler reward function.

# Development
A few key points of development are worth mentioning.

### State Features
As described in 
[the code](https://github.com/openai/gym/blob/39ba73ff17ce4df2ea08c1f9ab88f0726a476afb/gym/envs/mujoco/swimmer.py#L70-L79)
for this environment, the worm is characterized but the usual set of angles and velocities, all of which are in the 
range `[-infinity, +infinity]`. In this type of state space, the best action to take at a given time likely depends on 
all state variables at that time. When all variables are continuous as here, a straightforward way of encoding this 
dependence is to form a discrete category for each possible combination of state-variable signs. With an 
8-dimensional state, this results in `2^8=256` state categories. Only one of these categories is active at any 
particular time step, so we also call this a one-hot state category representation. The active category is associated
with the values of the state features (angles and velocities), and all inactive state categories receive zeros. This 
results in a 2049-dimensional state vector (`2^8 * 8` feature values, plus the intercept).

### Action Models
The swimming worm defines two actions corresponding to the continuous torques (each in `[-1, +1]`) applied at the two 
joints. We build a policy for these torques by modeling two beta distributions in terms of the 2049-dimensional state 
vector described above. Specifically, for each beta distribution (torque action), we model shape parameter `a` as a 
linear function of the 2049-dimensional state vector and shape parameter `b` as another linear function of the 
2049-dimensional state vector. Thus, there are 4098 parameters per action for a total of 8196 parameters in the complete
policy. The training episodes provide updates for these parameters via determinations of whether the actions taken 
performed better (i.e., swimming farther) or worse than estimated by a baseline state-value estimator. These better and 
worse determinations -- combined with the gradient of the beta distribution PDF with respect to the policy parameters
-- improve the policy over time. See the [continuous mountain car](mountain_car_continuous.md) for details of these 
calculations using JAX for automatic function differentiation.

### Baseline and Policy Gradient Learning Rates
The full argument list for training the agent is provided below. These arguments are quite similar to those used in 
other case studies for continuous control, except in one regard:  in other studies, setting a value of `--eta0` (the 
learning rate for the baseline state-value estimator) equal to `--alpha` (the learning rate along the policy gradient)
worked fine. In the case of the swimming worm, for reasons I have not identified, setting these learning rates to be 
equal never resulted in effective learning. Setting `--eta0` to be 1-2 orders of magnitude larger than `--alpha` did
work.

# Training
The following command trains an agent for the MuJoCo swimming worm environment using policy gradient optimization 
with a baseline state-value estimator:

```
rlai train --random-seed 12345 --agent rlai.policy_gradient.ParameterizedMdpAgent --gamma 1.0 --environment rlai.core.environments.gymnasium.Gym --gym-id Swimmer-v5 --render-every-nth-episode 500 --video-directory ~/Desktop/swimmer_videos --T 500 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 50000 --v-S rlai.state_value.function_approximation.ApproximateStateValueEstimator --feature-extractor rlai.core.environments.gymnasium.SignedCodingFeatureExtractor --function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_error --sgd-alpha 0.0 --learning-rate constant --eta0 0.001 --policy rlai.policy_gradient.policies.continuous_action.ContinuousActionBetaDistributionPolicy --policy-feature-extractor rlai.core.environments.gymnasium.SignedCodingFeatureExtractor --alpha 0.00001 --update-upon-every-visit True --num-episodes-per-checkpoint 500 --checkpoint-path ~/Desktop/swimmer_checkpoint.pickle --save-agent-path ~/Desktop/swimmer_agent.pickle --log DEBUG
```

The arguments are explained below.

### RLAI
* `train`:  Train the agent. 
* `--random-seed 12345`:  For reproducibility.

### Agent
* `--agent rlai.policy_gradient.ParameterizedMdpAgent`:  Standard parameterized MDP agent. 
* `--gamma 1.0`:  Do not discount.

### Environment
* `--environment rlai.core.environments.gymnasium.Gym`:  Environment class.
* `--gym-id Swimmer-v5`:  Gym environment identifier.
* `--render-every-nth-episode 500`:  Render a video every 500 episodes.
* `--video-directory ~/Desktop/swimmer_videos`:  Where to store rendered videos.
* `--T 500`:  Limit episodes to 500 steps.

### Training Function and Episodes
* `--train-function rlai.policy_gradient.monte_carlo.reinforce.improve`:  Run the REINFORCE policy gradient optimization
algorithm.
* `--num-episodes 50000`:  Run 50000 episodes.

### Baseline State-Value Estimator
* `--v-S rlai.state_value.function_approximation.ApproximateStateValueEstimator`:  Baseline state-value estimator.  
* `--feature-extractor rlai.core.environments.gymnasium.SignedCodingFeatureExtractor`:  Feature extractor for the
baseline state-value estimator.
* `--function-approximation-model rlai.models.sklearn.SKLearnSGD`:  Use SKLearn's SGD for the baseline state-value 
estimator.
* `--loss squared_error`:  Use a squared-error loss within the baseline state-value estimator.
* `--sgd-alpha 0.0`:  Do not use regularization.
* `--learning-rate constant`:  Use a constant learning rate schedule.
* `--eta0 0.001`:  Learning rate.

### Policy
* `--policy rlai.policy_gradient.policies.continuous_action.ContinuousActionBetaDistributionPolicy`:  Use the beta
distribution to model the action-density distribution within the policy.
* `--policy-feature-extractor rlai.core.environments.gymnasium.SignedCodingFeatureExtractor`:  Feature extractor
for the policy gradient optimizer.
* `--alpha 0.00001`:  Learning rate for policy gradient updates. See the results for details. This was set to 0.0001 
  for ~50,000 training episodes, at which point a degenerate policy was produced. Restarting from a reasonable policy 
  and `--alpha 0.00001` allowed the training to continue for ~20,000 training episodes without further issues.
* `--update-upon-every-visit True`:  Update the policy's action-density distribution every time a state is encountered
  (as opposed to the first visit only).

### Output
* `--num-episodes-per-checkpoint 500`:  Store a resumable checkpoint file every 500 episodes.
* `--checkpoint-path ~/Desktop/swimmer_checkpoint.pickle`:  Where to store checkpoint files (an index is inserted).
* `--save-agent-path ~/Desktop/swimmer_agent.pickle`:  Where to save the final agent.
* `~/Desktop/swimmer_agent.pickle --log DEBUG`:  Produce debug logging output.

# Results

The following sequence of videos shows the progression of swimmer policies.

### First Agent:  Random
This is the result on the first training episode, where the agent is purely random. 

{% include youtubePlayer.html id="ZeUbC9U9Zas" %}

### Scooping Segment
This is the result after several thousand training episodes. The agent has developed a scooping action with the front 
segment, but without much coordination of the back segment.

{% include youtubePlayer.html id="pnHlOhsRq6A" %}

### Improved Front Oscillations
Compared with the previous, this agent makes fuller oscillations of the front segment, but it still exhibits little 
coordination with the back segment.

{% include youtubePlayer.html id="sNRsMgv8FGI" %}

### Coordinated Segments
This is the first part of training in which clear coordination appears between the front and back segments, resulting in
smoother, faster swimming. 

{% include youtubePlayer.html id="tRwPN34U1fk" %}

### Failure
The policy collapsed at some point between the previous stage and this one, likely the result of a step size 
(`--alpha`) that was too large. Restarting just prior to this episode with a smaller step size permitted continued 
progress.

{% include youtubePlayer.html id="fjqUQm8iKXA" %}

### Final Agent
This is the final agent, which shows coordinated actuation of the front and back joints.

{% include youtubePlayer.html id="9a0jYLADr-c" %}

The agent seems effective, but it took ~70,000 episodes and about 3 days of runtime to train. See 
[here](mujoco_swimming_worm_pooled.md) for how I sped things up.