# MuJoCo Swimming Worm (Pooled Processes)
* Content
{:toc}

# Introduction
One hurdle in developing a control agent for the MuJoCo swimming worm (see [here](mujoco_swimming_worm.md)) was runtime.
It took ~70,000 training episodes and ~3 days of runtime to train the agent. Several factors contribute to the 
runtime:  policy dimensionality, learnings rates, the speed of simulation via `gym` and `mujoco-py`, and the speed of JAX to 
calculate policy gradients are a few that come to mind. Instead of working to mitigate these issues, I was curious to 
explore the use of multiple CPU cores during training. The original attempt ran on one of six available CPU cores, and 
Intel Hyper-Threading should extend the capacity beyond 6x. The question is, how should multiple CPU cores be 
leveraged? I was certain that this question had been answered many times in many ways, but as usual with my RLAI work
I wanted to find an answer without skipping ahead to known solutions.

# Concept
I tried several approaches that did not work.

1. I had each process/core work on its own subpool of agents, and periodically the processes would exchange their best
   known agents with each other. However, maintaining more agents than available CPU cores turned out to be a bad idea, 
   since any work on a suboptimal agent in a subpool was a distraction from improving the best agent in the subpool.
1. I then compounded the previous bad idea by introducing a low (epsilon) probability of a subpool process selecting -- 
   not the best agent from other pools -- but a random agent from across all subpools. This just slowed things down even
   further.
1. ...
1. ... (I tried several variations on the above without success.)
1. ...

Two things became clear:

1. Each process should attempt to improve the best known agent for a short amount of time. That is, the picture should 
   always look as follows:
   
1. It wasn't clear how the individual processes should report back on the performance of the agent they had obtained. 
   The agent accumulates rewards during each training episode; however, the agent's is updated immediately upon episode 
   completion. So the accumulated reward does not correspond exactly to the agent in hand. Even if it did, the rewards 
   are only from a single episode and are thus prone to being noisy. An additional evaluation phase is needed during 
   which the policy is held constant. The final concept is shown below:
   
# Implementation

# Training

# Results

2.45 minutes per pool iteration for 400 iterations gives a total runtime of ~16 hours. This is a 78% reduction compared 
with the previous attempt's runtime of ~72 hours.

![results](mujoco_worm_pooled.png)