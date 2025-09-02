"""
Proximal Policy Optimization
(https://arxiv.org/pdf/1707.06347)

Implementation of PPO using jax.

PPO is a policy gradient method that prevents destructive policy upates. 
It uses surrogate objective with a clipping mechanism. 
The policy is parameterized by a neural network that outputs action probabilities. 
A separate value network estimates state values for advantage calculation.
"""
