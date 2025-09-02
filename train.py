import jax
from jax._src.numpy.lax_numpy import _convert_to_array_if_dtype_fails
import jax.numpy as jnp
from jax import random
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from ppo import PPOAgent, PPOConfig, RolloutData, compute_gae


def collect_rollout(agent, env, key, n_steps):
    """Colloect rollout with gae bootstrap"""
    obs, _ = env.reset()
    obs = jnp.array(obs)

    observations, actions, rewards, dones, values, log_probs = [], [], [], [], [], []

    for step in range(n_steps):
        key, action_key = random.split(key)
        action, log_prob, value = agent.get_action(obs, action_key)

        observations.append(obs)
        actions.append(action)
        values.append(value)
        log_probs.append(log_prob)

        # env step
        action_np = np.array(action)
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        rewards.append(reward)
        dones.append(done)
        obs = jnp.array(next_obs)

        if done:
            obs, _ = env.reset()
            obs = jnp.array(obs)

    # get final value for gae bootstrap
    key, final_key = random.split(key)
    _, _, next_value = agent.get_action(obs, final_key)

    # convert to arrays
    observations = jnp.array(observations)
    actions = jnp.array(actions)
    rewards = jnp.array(rewards)
    dones = jnp.array(dones)
    values = jnp.array(values)
    log_probs = jnp.array(log_probs)

    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        float(next_value),
        agent.config.gamma,
        agent.config.gae_lambda,
    )

    return RolloutData(
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        values=values,
        log_probs=log_probs,
        advantages=advantages,
        returns=returns,
        next_value=float(next_value),
    )


def evaluate_policy(agent, env, key, n_eps=5):
    """Evaluate current policy"""
    total_returns = []

    for episode in range(n_eps):
        obs, _ = env.reset()
        obs = jnp.array(obs)

        episode_return = 0
        done = False
        max_steps = 1000
        step = 0

        while not done and step < max_steps:
            action, _, _ = agent.get_action(obs, key, deterministic=True)
            action_np = np.array(action)

            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            obs = jnp.array(obs)
            episode_return += reward
            step += 1

        total_returns.append(episode_return)

    return float(np.mean(total_returns))


def train_ppo(env_name="HalfCheetah-v5", total_timesteps=500_000):
    """Training ppo"""

    # hyperparameters
    config = PPOConfig(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        entropy_coeff=0.01,
        clip_epsilon=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        value_coeff=0.5,
        max_grad_norm=0.5,
    )

    # env
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"PPO Training on {env_name}")
    print(f"Obs: {obs_dim}, Actions: {action_dim}")
    print(
        f"Budget: {total_timesteps:,} timesteps ({
            total_timesteps // config.n_steps
        } updates)"
    )
    print(f"Config: lr={config.learning_rate}, entropy={config.entropy_coeff}")

    # agent
    agent = PPOAgent(obs_dim, action_dim, config)

    # training storage
    key = random.PRNGKey(42)
    episode_returns = []
    timesteps_elapsed = []

    training_logs = {
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clipped_fraction": [],
        "explained_variance": [],
    }

    n_updates = total_timesteps // config.n_steps
    best_return = float("-inf")

    print(f"\nStarting training...")

    for update in tqdm(range(n_updates), desc="ppo training"):
        # collect rollout
        key, rollout_key = random.split(key)
        rollout_data = collect_rollout(agent, env, rollout_key, config.n_steps)

        # update policy
        metrics = agent.update(rollout_data)

        # log training metrics
        for metric_name in training_logs.keys():
            training_logs[metric_name].append(metrics[metric_name])

        # evaluate periodically
        if update % 10 == 0:
            key, eval_key = random.split(key)
            avg_return = evaluate_policy(agent, env, eval_key)
            episode_returns.append(avg_return)
            timesteps_elapsed.append(update * config.n_steps)

            if avg_return > best_return:
                best_return = avg_return
                status = "YAY! NEW BEST LESSGOO!!"

            else:
                status = "NO CHANGE"

            print(
                f"{status} Update {update:3d} ({update * config.n_steps:6,} steps): "
                f"Return={avg_return:7.1f}, "
                f"Actor={metrics['actor_loss']:.3f}, "
                f"Critic={metrics['critic_loss']:.3f}, "
                f"Entropy={metrics['entropy']:.2f}, "
                f"KL={metrics['approx_kl']:.4f}"
            )

    env.close()

    # plotting
    plot_results(
        episode_returns, timesteps_elapsed, training_logs, env_name, best_return
    )

    return agent, episode_returns, training_logs


def plot_results(returns, timesteps, logs, env_name, best_return):
    """Plot training results"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"PPO Training Results - {env_name}", fontsize=16, fontweight="bold")

    # learning curve
    axes[0, 0].plot(timesteps, returns, "b-", linewidth=2, alpha=0.8)
    axes[0, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5, label="Zero reward")
    axes[0, 0].axhline(
        y=best_return,
        color="g",
        linestyle=":",
        alpha=0.7,
        label=f"Best: {best_return:.1f}",
    )
    axes[0, 0].set_title("Learning Curve")
    axes[0, 0].set_xlabel("Timesteps")
    axes[0, 0].set_ylabel("Episode Return")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # add success indicator
    if max(returns) > 0:
        axes[0, 0].text(
            0.02,
            0.98,
            "Positive rewards achieved!",
            transform=axes[0, 0].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
        )

    # actor and critic Loss
    axes[0, 1].plot(logs["actor_loss"], "r-", alpha=0.7, label="Actor Loss")
    axes[0, 1].plot(logs["critic_loss"], "g-", alpha=0.7, label="Critic Loss")
    axes[0, 1].set_title("Loss Curves")
    axes[0, 1].set_xlabel("Update")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # entropy
    axes[0, 2].plot(logs["entropy"], "purple", linewidth=2)
    axes[0, 2].set_title("Policy Entropy")
    axes[0, 2].set_xlabel("Update")
    axes[0, 2].set_ylabel("Entropy")
    axes[0, 2].grid(True, alpha=0.3)

    # KL divergence
    axes[1, 0].plot(logs["approx_kl"], "orange", linewidth=2)
    axes[1, 0].axhline(y=0.01, color="r", linestyle="--", alpha=0.5, label="Target KL")
    axes[1, 0].set_title("KL Divergence")
    axes[1, 0].set_xlabel("Update")
    axes[1, 0].set_ylabel("Approx KL")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # clipped fraction
    axes[1, 1].plot(logs["clipped_fraction"], "brown", linewidth=2)
    axes[1, 1].set_title("PPO Clipping")
    axes[1, 1].set_xlabel("Update")
    axes[1, 1].set_ylabel("Clipped Fraction")
    axes[1, 1].grid(True, alpha=0.3)

    # explained variance
    axes[1, 2].plot(logs["explained_variance"], "teal", linewidth=2)
    axes[1, 2].axhline(y=0, color="r", linestyle="--", alpha=0.5)
    axes[1, 2].set_title("Value Function Quality")
    axes[1, 2].set_xlabel("Update")
    axes[1, 2].set_ylabel("Explained Variance")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # env
    env_name = "HalfCheetah-v5"

    try:
        agent, returns, logs = train_ppo(env_name, total_timesteps=200_000)

        print(f"\nTraining Summary:")
        print(f"Final return: {returns[-1]:.1f}")
        print(f"Best return: {max(returns):.1f}")
        print(f"Improvement: {returns[-1] - returns[0]:+.1f}")

        positive_returns = [r for r in returns if r > 0]
        if positive_returns:
            print(f"Achieved {len(positive_returns)} positive evaluations!")
            print(f"Best positive return: {max(positive_returns):.1f}")
        else:
            print(f"No positive returns yet. Consider:")
            print(f" - Longer training (increase total_timesteps)")
            print(f" - Try Hopper-v5 (easier environment)")

    except Exception as e:
        print(f"Error: {e}")

