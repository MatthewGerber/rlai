# Inverted Pendulum
The inverted pendulum is also known as cart-pole balancing, where the goal is to keep a bottom-hinged pole balanced for
as long as possible. Imagine balancing a broom with the handle's end in your open palm. You can read more about this 
environment [here](https://gym.openai.com/envs/CartPole-v1/). Below is an example of running a random (untrained) agent 
in this environment. The episode terminates almost immediately as the agent loses balance control.

{% include youtubePlayer.html id="rGnf9CFwD7M" %}

## Training

Train a control agent for the inverted pendulum environment with the following command.
```
rlai_train --train --agent "rlai.agents.mdp.StochasticMdpAgent" --continuous-state-discretization-resolution 0.1 --gamma 1 --environment "rlai.environments.openai_gym.Gym" --gym-id "CartPole-v1" --render-every-nth-episode 5000 --video-directory "~/Desktop/cartpole_videos" --train-function "rlai.gpi.temporal_difference.iteration.iterate_value_q_pi" --mode "Q_LEARNING" --num-improvements 5000 --num-episodes-per-improvement 50 --T 1000 --epsilon 0.01 --make-final-policy-greedy True --num-improvements-per-plot 100 --num-improvements-per-checkpoint 100 --checkpoint-path "~/Desktop/cartpole_checkpoint.pickle" --save-agent-path "~/Desktop/cartpole_agent.pickle"
```

Arguments are explained below.
* `--train`:  Train the agent.
* `--agent "rlai.agents.mdp.StochasticMdpAgent"`:  Standard stochastic MDP agent. 
* `--continuous-state-discretization-resolution 0.1`:  Discretize the continuous state space into discrete intervals 
  with resolution 0.1. The methods used here are for discrete-state problems, so some type of discretization of the 
  continuous state space is required.
* `--gamma 1`:  No discount. All state-action pairs in the episode will receive equal credit for the total duration of 
  balancing achieved. 
* `--environment "rlai.environments.openai_gym.Gym"`:  Environment class. 
* `--gym-id "CartPole-v1"`:  OpenAI Gym environment identifier.
* `--render-every-nth-episode 5000`:  Render a video every 5000 episodes (100 improvements).
* `--video-directory "~/Desktop/cartpole_videos"`:  Where to store rendered videos.
* `--train-function "rlai.gpi.temporal_difference.iteration.iterate_value_q_pi"`:  Run iterative temporal-differencing 
  on the agent's state-action value function. 
* `--mode "Q_LEARNING"`:  Use q-learning to bootstrap the value of the next state-action pair. 
* `--num-improvements 5000`:  Number of policy improvements (iterations).
* `--num-episodes-per-improvement 50`:  Number of episodes per improvement.
* `--T 1000`:  Maximum number of time steps per episode. Without this, the episode length will be unconstrained. This 
  can slow down learning in later iterations where the agent has developed a reasonable policy.
* `--epsilon 0.01`:  Probability of behaving randomly at each time step.
* `--make-final-policy-greedy True`:  After all learning iterations, make the final policy greedy (i.e., `epsilon=0.0`).
* `--num-improvements-per-plot 100`:  Plot training performance every 100 iterations.
* `--num-improvements-per-checkpoint 100`:  Checkpoint the learning process every 100 iterations, enabling resumption.
* `--checkpoint-path "~/Desktop/cartpole_checkpoint.pickle"`:  Where to store the checkpoint.
* `--save-agent-path "~/Desktop/cartpole_agent.pickle"`:  Where to save the final agent.

The training progression is shown below.

![inverted-pendulum](https://github.com/MatthewGerber/rlai/raw/master/trained_agents/cartpole/cartpole_training.png)

In the left sub-figure above, the left y-axis shows how long the control agent is able to keep the pole balanced, the 
right y-axis shows the size of the state space, and the x-axis shows improvement iterations for the agent's policy. The 
right sub-figure shows the same reward y-axis but along a time x-axis. Based on the learning trajectory, it seems clear
that the agent would have continued to improve its policy given more time; however, as shown below, the results are 
quite satisfactory after 7 hours of wallclock training time.

## Results

The video below shows the trained agent controlling the inverted pendulum. Note how the agent actively controls both the 
vertical balance of the pendulum (ideally upright) and the cart's horizontal position along the track (ideally in the 
middle). The episode ends after the maximum number of time steps is reached, rather than due to a control failure.

{% include youtubePlayer.html id="bnQFT31_WfI" %}
