
"""
Logger for recording training statistics.
"""

import os
import json
import yaml
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter


def setup_logging(args, run_name) -> SummaryWriter:
    """
    Setup logging for log statistics
    Args:
        args: Command-line arguments containing hyperparameters.
        run_name (str): Name of the current run for logging purposes.
    Returns:
        SummaryWriter: TensorBoard summary writer for logging statistics.
    """

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    logdir = f"runs/{run_name}"
    print(f"Logging to {logdir}")
    os.makedirs(logdir, exist_ok=True)

    with open(f"{logdir}/config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(vars(args), f)

    print("Configs:")
    print(json.dumps(dict(sorted(vars(args).items())), indent=2))


    return writer


def log_stats(infos, writer, global_step, return_deque, length_deque):
    """
    Log statistics to TensorBoard during reinforcement learning.
    Args:
        infos (dict): Information dictionary containing episode statistics.
        writer (SummaryWriter): TensorBoard summary writer.
        global_step (int): Global environment step.
        return_deque (deque): Deque to store episodic returns.
        length_deque (deque): Deque to store episodic lengths.
    """
    if "final_info" in infos:
        for info in infos["final_info"]:
            if info and "episode" in info:
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                return_deque.extend(info["episode"]["r"])
                length_deque.extend(info["episode"]["l"])


def log_training_metrics(
    writer,
    optimizer,
    iteration,
    global_step,
    start_time,
    next_log,
    v_loss,
    pg_loss,
    entropy_loss,
    old_approx_kl,
    approx_kl,
    clipfracs,
    explained_var,
    time_collection,
    time_preprocessing,
    time_update,
    rewards,
    per_env_rewards_list,
    b_advantages,
    b_returns,
    b_logprobs,
    b_values,
    baseline_vals,
    grad_norms,
    policy_ratios,
    args,
):
    """
    Log training metrics to TensorBoard during reinforcement learning.

    Records optimization losses, policy statistics, timing information,
    reward statistics, advantage and return distributions, value function
    statistics, gradient norms, and policy ratio metrics for monitoring
    training progress.

    Args:
        writer (SummaryWriter): TensorBoard summary writer.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        iteration (int): Current training iteration.
        global_step (int): Global environment step.
        start_time (float): Training start time used for computing SPS.
        next_log (int): Next scheduled logging step.
        v_loss (torch.Tensor): Value function loss.
        pg_loss (torch.Tensor): Policy gradient loss.
        entropy_loss (torch.Tensor): Entropy regularization loss.
        old_approx_kl (torch.Tensor): Previous approximate KL divergence.
        approx_kl (torch.Tensor): Current approximate KL divergence.
        clipfracs (list[float]): PPO clipping fractions collected during updates.
        explained_var (float): Explained variance of the value function.
        time_collection (float): Time spent collecting rollouts.
        time_preprocessing (float): Time spent preprocessing rollouts.
        time_update (float): Time spent updating the policy.
        rewards (torch.Tensor): Collected rewards from rollout mode.
        per_env_rewards_list (list): Per-environment rewards in episode mode.
        b_advantages (torch.Tensor): Advantage estimates.
        b_returns (torch.Tensor): Computed returns.
        b_logprobs (torch.Tensor): Log probabilities of sampled actions.
        b_values (torch.Tensor): Value function predictions.
        baseline_vals (torch.Tensor | None): Baseline estimates when applicable.
        grad_norms (list[float]): Gradient norms recorded during optimization.
        policy_ratios (list[float]): Policy ratios recorded during optimization.
        args: Command-line arguments containing training configuration.

    Returns:
        int: Updated value of ``next_log``.
    """
    # TRY NOT TO MODIFY: record rewards for plotting purposes
    if args.log_every == 0 or global_step >= next_log:
        if args.log_every > 0:
            next_log = global_step + args.log_every
        
        with torch.no_grad():
            # Charts and losses
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/updates", iteration, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            
            # Timing metrics
            writer.add_scalar("time/collection", time_collection, global_step)
            writer.add_scalar("time/preprocessing", time_preprocessing, global_step)
            writer.add_scalar("time/update", time_update, global_step)
            writer.add_scalar("time/total_iteration", time_collection + time_preprocessing + time_update, global_step)
            
            # Reward statistics
            if args.num_steps > 0:
                rewards_flat = rewards.flatten()
                writer.add_scalar("stats/rewards_mean", rewards_flat.mean().item(), global_step)
                writer.add_scalar("stats/rewards_std", rewards_flat.std().item(), global_step)
                writer.add_scalar("stats/rewards_min", rewards_flat.min().item(), global_step)
                writer.add_scalar("stats/rewards_max", rewards_flat.max().item(), global_step)
            else:
                # Episode mode: compute reward statistics from per_env_rewards
                all_rewards = [r for env_rews in per_env_rewards_list for r in env_rews]
                if len(all_rewards) > 0:
                    writer.add_scalar("stats/rewards_mean", np.mean(all_rewards), global_step)
                    writer.add_scalar("stats/rewards_std", np.std(all_rewards), global_step)
                    writer.add_scalar("stats/rewards_min", np.min(all_rewards), global_step)
                    writer.add_scalar("stats/rewards_max", np.max(all_rewards), global_step)
            
            # Advantage, return, and baseline statistics
            writer.add_scalar("stats/advantages_mean", b_advantages.mean().item(), global_step)
            writer.add_scalar("stats/advantages_std", b_advantages.std().item(), global_step)
            writer.add_scalar("stats/advantages_min", b_advantages.min().item(), global_step)
            writer.add_scalar("stats/advantages_max", b_advantages.max().item(), global_step)
            writer.add_scalar("stats/returns_mean", b_returns.mean().item(), global_step)
            writer.add_scalar("stats/returns_std", b_returns.std().item(), global_step)
            writer.add_scalar("stats/returns_min", b_returns.min().item(), global_step)
            writer.add_scalar("stats/returns_max", b_returns.max().item(), global_step)
            writer.add_scalar("stats/logprobs_mean", b_logprobs.mean().item(), global_step)
            writer.add_scalar("stats/logprobs_std", b_logprobs.std().item(), global_step)
            if args.use_value_fn:
                writer.add_scalar("stats/values_mean", b_values.mean().item(), global_step)
                writer.add_scalar("stats/values_std", b_values.std().item(), global_step)
            if args.return_type != "gae" and 'baseline_vals' in locals():
                writer.add_scalar("stats/baseline_mean", baseline_vals.mean().item(), global_step)
                writer.add_scalar("stats/baseline_std", baseline_vals.std().item(), global_step)
            
            # Gradient statistics
            writer.add_scalar("stats/grad_norm_mean", np.mean(grad_norms), global_step)
            writer.add_scalar("stats/grad_norm_std", np.std(grad_norms), global_step)
            writer.add_scalar("stats/grad_norm_max", np.max(grad_norms), global_step)
            writer.add_scalar("stats/grad_norm_min", np.min(grad_norms), global_step)
            
            # Policy ratio statistics (indicates how much policy is changing)
            writer.add_scalar("stats/policy_ratio_mean", np.mean(policy_ratios), global_step)
            writer.add_scalar("stats/policy_ratio_std", np.std(policy_ratios), global_step)
    