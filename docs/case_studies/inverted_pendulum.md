# Inverted Pendulum
The inverted pendulum is also known as cart-pole balancing. You can read more about this environment 
[here](https://gym.openai.com/envs/CartPole-v1/). Below is an example of running a random (untrained) untrained agent in 
this environment:

{% include youtubePlayer.html id="rGnf9CFwD7M" %}

## Training

To train a control agent for the inverted pendulum environment, first install the `rlai` package and then run the 
following command.
```
rlai_train --train --agent "rlai.agents.mdp.StochasticMdpAgent" --continuous-state-discretization-resolution 0.1 --gamma 1 --environment "rlai.environments.openai_gym.Gym" --gym-id "CartPole-v1" --render-every-nth-episode 5000 --video-directory "~/Desktop/cartpole_videos" --train-function "rlai.gpi.temporal_difference.iteration.iterate_value_q_pi" --mode "Q_LEARNING" --num-improvements 5000 --num-episodes-per-improvement 50 --T 1000 --epsilon 0.01 --make-final-policy-greedy True --num-improvements-per-plot 100 --num-improvements-per-checkpoint 100 --checkpoint-path "~/Desktop/cartpole_checkpoint.pickle" --save-agent-path "~/Desktop/cartpole_agent.pickle"
```

Arguments are explained below.
```
--train 
--agent "rlai.agents.mdp.StochasticMdpAgent" 
--continuous-state-discretization-resolution 0.1 
--gamma 1 
--environment "rlai.environments.openai_gym.Gym" 
--gym-id "CartPole-v1" 
--render-every-nth-episode 5000 
--video-directory "~/Desktop/cartpole_videos" 
--train-function "rlai.gpi.temporal_difference.iteration.iterate_value_q_pi" 
--mode "Q_LEARNING" 
--num-improvements 5000 
--num-episodes-per-improvement 50 
--T 1000 
--epsilon 0.01 
--make-final-policy-greedy True 
--num-improvements-per-plot 100 
--num-improvements-per-checkpoint 100 
--checkpoint-path "~/Desktop/cartpole_checkpoint.pickle" 
--save-agent-path "~/Desktop/cartpole_agent.pickle"
```

The training progression is shown below.

![inverted-pendulum](https://github.com/MatthewGerber/rlai/raw/master/trained_agents/cartpole/cartpole_training.png)

In the left sub-figure above, the left y-axis shows how long the control agent is able to keep the pole balanced, and 
the right y-axis shows the size of the state space. The right sub-figure shows the same reward y-axis but along a time
x-axis.


## Results

The video below shows the trained agent controlling the inverted pendulum.

{% include youtubePlayer.html id="bnQFT31_WfI" %}
