"""
Proximal Policy Optimization
(https://arxiv.org/pdf/1707.06347)

Implementation of PPO using pure jax.

PPO is a policy gradient method that prevents destructive policy upates.
It uses surrogate objective with a clipping mechanism.
The policy is parameterized by a neural network that outputs action probabilities.
A separate value network estimates state values for advantage calculation.
"""

from typing import NamedTuple, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import random, grad, jit
from numpy._typing import _128Bit


@dataclass
class PPOConfig:
    """Hyperparams for PPO Algo"""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 64
    n_steps: int = 2048


class RolloutData(NamedTuple):
    """Data collected during env rollouts"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    values: jnp.ndarray
    log_probs: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray
    next_value: float


def init_network_params(
    key: jax.Array, obs_dim: int, action_dim: int, hidden_dim: int = 64
):
    """Init neural network"""
    k1, k2, k3, k4, k5 = random.split(key, 5)

    # shared layers
    w1 = random.normal(k1, (obs_dim, hidden_dim)) * jnp.sqrt(2.0 / obs_dim)
    b1 = jnp.zeros(hidden_dim)
    w2 = random.normal(k2, (hidden_dim, hidden_dim)) * jnp.sqrt(2.0 / hidden_dim)
    b2 = jnp.zeros(hidden_dim)

    # actor head (small inits for stability)
    w_actor = random.normal(k3, (hidden_dim, action_dim)) * 0.01
    b_actor = jnp.zeros(action_dim)

    # actor log std
    # initial std = 0.37. starting with low std for stability
    log_std = jnp.ones(action_dim) * -1.0

    # critic head
    w_critic = random.normal(k4, (hidden_dim, 1)) * 0.01
    b_critic = jnp.zeros(1)

    return {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "w_actor": w_actor,
        "b_actor": b_actor,
        "log_std": log_std,
        "w_critic": w_critic,
        "b_critic": b_critic,
    }


def forward_network(params, obs):
    """Forward pass thru the network"""
    x = jnp.dot(obs, params["w1"]) + params["b1"]
    x = jnp.tanh(x)
    x = jnp.dot(x, params["w2"]) + params["b2"]
    x = jnp.tanh(x)

    # actor head
    action_mean = jnp.dot(x, params["w_actor"]) + params["b_actor"]
    action_log_std = params["log_std"]

    # critic head
    value = jnp.dot(x, params["w_critic"]) + params["b_critic"]
    value = jnp.squeeze(value)

    return action_mean, action_log_std, value


def sample_action(
    key: jax.Array, action_mean: jnp.ndarray, action_log_std: jnp.ndarray
):
    """Sample action from gaussian policy"""
    action_std = jnp.exp(action_log_std)
    action = action_mean + action_std * random.normal(key, action_mean.shape)

    # log prob calculated using the pdf of gaussian dist
    log_prob = -0.5 * (
        jnp.sum((action - action_mean) ** 2 / (action_std**2))
        + jnp.sum(2 * action_log_std)
        + action_mean.shape[0] * jnp.log(2 * jnp.pi)
    )

    return action, log_prob


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    next_value: float,
    gamma: float,
    # GAE smoothing parameter, controls the bias variance tradeoff in advantage estimation
    gae_lambda: float,
):
    T = len(rewards)  # total timesteps
    advantages = jnp.zeros_like(rewards)

    # starting with bootstrap value for the final step
    last_advantage = 0.0

    # work backwards from T-1 to 0, GAE depends on future values (bootstrap estimation)
    for t in reversed(range(T)):
        if t == T - 1:
            # last step: use next_value for bootstrap
            next_v = next_value
        else:
            # earlier steps: use next_value from trajectory
            next_v = values[t + 1]

        # compute td error
        delta = rewards[t] + gamma * next_v * (1.0 - dones[t]) - values[t]

        # compute advantage with GAE
        advantages = advantages.at[t].set(
            delta + gamma * gae_lambda * (1.0 - dones[t]) - values[t]
        )

        last_advantage = advantages[t]

    # returns are advantages + values
    returns = advantages + values

    # normalizing the advantages for stability
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    return advantages, returns


def ppo_loss_fn(params, batch: RolloutData, clip_epsilon, value_coeff, entropy_coeff):
    """Compute PPO Loss"""
    # forward pass
    action_mean, action_log_std, values = forward_network(params, batch.observations)

    # current policy log probs
    action_std = jnp.exp(action_log_std)
    # calculated using the gaussian log likelihood formula
    current_log_probs = -0.5 * (
        jnp.sum((batch.actions - action_mean) ** 2 / (action_std**2), axis=-1)
        + jnp.sum(2 * action_log_std)
        + batch.actions.shape[-1] * jnp.log(2 * jnp.pi)
    )

    # ppo actor loss
    ratio = jnp.exp(current_log_probs - batch.log_probs)
    clipped_ratio = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    actor_loss = -jnp.mean(
        jnp.minimum(ratio * batch.advantages, clipped_ratio * batch.advantages)
    )

    # critic loss (MSE)
    critic_loss = jnp.mean((values - batch.returns) ** 2)

    # entropy bonus
    # entropy is a measure of randomness or uncertainty in the policy's action distribution
    entropy = jnp.sum(action_log_std) + 0.5 * batch.actions.shape[-1] * (
        1 + jnp.log(2 * jnp.pi)
    )
    entropy_loss = -entropy_coeff * entropy

    # total loss
    total_loss = actor_loss + value_coeff * critic_loss + entropy_loss

    # debugging metrics
    approx_kl = jnp.mean(batch.log_probs - current_log_probs)
    approx_kl = jnp.mean(batch.log_probs - current_log_probs)
    clipped_fraction = jnp.mean(jnp.abs(ratio - 1.0) > clip_epsilon)
    explained_variance = 1.0 - jnp.var(batch.returns - values) / jnp.var(batch.returns)

    metrics = {
        "total_loss": total_loss,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clipped_fraction": clipped_fraction,
        "explained_variance": explained_variance,
    }

    return total_loss, metrics


def clip_gradients(grads, max_norm):
    """Clip grads by global norm"""
    global_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    scale = jnp.minimum(max_norm / (global_norm + 1e-6), 1.0)
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


def adam_update(params, grads, opt_state, learning_rate):
    """Adam optimizer update"""
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    if opt_state is None:
        opt_state = {
            "m": jax.tree_util.tree_map(jnp.zeros_like, params),
            "v": jax.tree_util.tree_map(jnp.zeros_like, params),
            "t": 0,
        }

    opt_state["t"] += 1

    # update moments
    opt_state["m"] = jax.tree_util.tree_map(
        lambda m, g: beta1 * m + (1 - beta1) * g, opt_state["m"], grads
    )
    opt_state["v"] = jax.tree_util.tree_map(
        lambda v, g: beta2 * v + (1 - beta2) * g * g, opt_state["v"], grads
    )

    # bias correction
    m_hat = jax.tree_util.tree_map(
        lambda m: m / (1 - beta1 ** opt_state["t"]), opt_state["m"]
    )
    v_hat = jax.tree_util.tree_map(
        lambda v: v / (1 - beta2 ** opt_state["t"]), opt_state["v"]
    )

    # parameter update
    new_params = jax.tree_util.tree_map(
        lambda p, m, v: p - learning_rate * m / (jnp.sqrt(v) + eps),
        params,
        m_hat,
        v_hat,
    )

    return new_params, opt_state


class PPOAgent:
    """PPO agent for continuous control"""

    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig = None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or PPOConfig()

        # init network params
        key = random.PRNGKey(42)
        self.params = init_network_params(key, obs_dim, action_dim)
        self.opt_state = None

        # jit compile for speed
        self.jit_forward = jit(forward_network)
        self.grad_fn = jit(grad(ppo_loss_fn, has_aux=True))

    def get_action(self, obs: jnp.ndarray, key: jax.Array, deterministic: bool = False):
        """Get action from policy"""
        action_mean, action_log_std, value = self.jit_forward(self.params, obs)

        if deterministic:
            action = action_mean
            log_prob = jnp.zeros(())
        else:
            action, log_prob = sample_action(key, action_mean, action_log_std)

        return action, log_prob, value

    def update(self, rollout_data: RolloutData):
        "Update policy using PPO"
        total_samples = len(rollout_data.observations)
        indices = jnp.arange(total_samples)

        all_metrics = []

        for epoch in range(self.config.n_epochs):
            # shuffle data
            key = random.PRNGKey(epoch)
            indices = random.permutation(key, indices)

            # train on mini batches
            for start in range(0, total_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, total_samples)
                batch_indices = indices[start:end]

                batch = RolloutData(
                    observations=rollout_data.observations[batch_indices],
                    actions=rollout_data.actions[batch_indices],
                    rewards=rollout_data.rewards[batch_indices],
                    dones=rollout_data.dones[batch_indices],
                    values=rollout_data.values[batch_indices],
                    log_probs=rollout_data.log_probs[batch_indices],
                    advantages=rollout_data.advantages[batch_indices],
                    returns=rollout_data.returns[batch_indices],
                    next_value=rollout_data.next_value,
                )

                # compute gradients
                grads, metrics = self.grad_fn(
                    self.params,
                    batch,
                    self.config.clip_epsilon,
                    self.config.value_coeff,
                    self.config.entropy_coeff,
                )

                # clip gradients
                grads = clip_gradients(grads, self.config.max_grad_norm)

                # update parameters
                self.params, self.opt_state = adam_update(
                    self.params, grads, self.opt_state, self.config.learning_rate
                )

                all_metrics.append(metrics)

        # avg metrics over all updates
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = float(jnp.mean(jnp.array([m[key] for m in all_metrics])))

        return avg_metrics
