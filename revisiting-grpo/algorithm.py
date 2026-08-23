# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import collections
import json
import random
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.rl_agent import Agent
from src.args import add_args
from src.utils.wandb_utils import init_wandb
from src.checkpoints import save_checkpoint, load_checkpoint
from src.utils.logger import setup_logging, log_training_metrics
from src.training import training_minibatch
from src.rollout import collect_fixed_rollout, collect_episode_rollout, flatten_rollout_batch
from src.returns import calculate_gae_returns, calculate_td_returns, calculate_mc_returns
import shutil

from environment import make as make_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    add_args(parser)
    args = parser.parse_args()

    # ========== CONFIGURATION VALIDATION ==========
    # This script supports 3 main modes based on return_type:
    #
    # 1. GAE (Generalized Advantage Estimation) - Standard PPO
    #    --return-type gae --num-steps N (where N > 0)
    #    --baseline-type is IGNORED (GAE computes advantages directly)
    #    Uses: GAE with lambda for advantage estimation, N-step rollouts
    #    Example: --return-type gae --num-steps 128
    #
    # 2. TD-n (Temporal Difference with n-step returns) - PPO/GRPO variant
    #    --return-type td --num-steps N (where N > 0)
    #    --baseline-type determines how advantages are centered (value, constant, uniform, stats, batch_mean, ema, same_seed_mean)
    #    Uses: n-step TD returns (where n=num_steps), then subtracts baseline
    #    Example: --return-type td --num-steps 128 --baseline-type value
    #    Example: --return-type td --num-steps 5 --baseline-type same_seed_mean --num-groups 4
    #
    # 3. MC (Monte Carlo) - GRPO with full episode returns
    #    --return-type mc --num-steps 0
    #    --baseline-type determines how advantages are centered (value, constant, uniform, stats, batch_mean, ema, same_seed_mean)
    #    Uses: Full episode returns, then subtracts baseline
    #    Example: --return-type mc --num-steps 0 --baseline-type batch_mean
    #    Example: --return-type mc --num-steps 0 --baseline-type same_seed_mean --num-groups 4
    #
    # Additional baseline requirements:
    # - same_seed_mean requires --num-groups > 0
    # - constant requires --baseline-constant to be set
    # - uniform uses --baseline-uniform-low and --baseline-uniform-high (auto-inferred from env if not set)
    # - ema uses --baseline-ema-beta (default 0.9)

    # Validate return_type and num_steps configuration
    if args.return_type == "gae":
        assert args.num_steps > 0, \
            "GAE requires num_steps > 0. Use --num-steps N where N > 0."
    elif args.return_type == "td":
        assert args.num_steps > 0, \
            "TD requires num_steps > 0 (num_steps is used as n in TD-n). Use --num-steps N where N > 0."
    elif args.return_type == "mc":
        if args.num_steps != 0:
            print("WARNING: Using num_steps != 0 with --return-type mc. Monte Carlo requires num_steps = 0 (full episodes) for correctness.")
    
    # Validate baseline-specific requirements
    if args.baseline_type == "same_seed_mean":
        assert args.num_groups > 0, \
            "same_seed_mean baseline requires --num-groups > 0 to define environment groups."
        assert args.num_envs % args.num_groups == 0, \
            f"num_envs ({args.num_envs}) must be divisible by num_groups ({args.num_groups}) for same_seed_mean baseline."
    
    if args.baseline_type == "constant":
        assert args.baseline_constant is not None, \
            "constant baseline requires --baseline-constant to be set."
    
    # Validate value function usage
    if (args.return_type == "gae" or args.baseline_type in ("value", "td")) and not args.use_value_fn:
        print(f"WARNING: --baseline-type {args.baseline_type} is set but --use-value-fn is False. The value function won't be trained.")

    # Batch/iteration sizing
    if args.num_steps > 0:
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(max(1, args.batch_size // args.num_minibatches))
        args.num_iterations = max(1, args.total_timesteps // max(1, args.batch_size))
        assert args.batch_size >= args.num_minibatches, \
            f"batch_size ({args.batch_size}) must be >= num_minibatches ({args.num_minibatches}). Increase num_steps or num_envs, or decrease num_minibatches."
    else:
        # Episode mode: dynamic batch size per iteration, stop by total timesteps
        args.batch_size = 0
        args.minibatch_size = 0
        args.num_iterations = max(1, args.total_timesteps)

    if args.num_checkpoints > 0:
        args.checkpoint_every = args.total_timesteps // args.num_checkpoints
        print(f"Setting checkpoint every {args.checkpoint_every} steps")
    args.checkpoint_param_filters = json.loads(args.checkpoint_param_filters) if args.checkpoint_param_filters else {}

    if args.run_name:
        run_name = args.run_name
        args.exp_name = "__".join(args.run_name.split("__")[:-1])
    else:
        run_name = f"{args.exp_name}__{args.env_id.replace('/', '_').replace('-', '_').lower()}__{args.seed}"

    if os.path.exists(f"runs/{run_name}/config.yaml"):
        if args.overwrite:
            print(f"Run directory {run_name} already exists. Overwriting.")
            shutil.rmtree(f"runs/{run_name}")
        else:
            print(f"Run directory {run_name} already exists. Exiting.")
            exit(0)


    # Initialize wandb if tracking is enabled

    init_wandb(args, run_name)

    writer = setup_logging(args, run_name)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if torch.backends.mps.is_available() and args.cuda:
        device = torch.device("mps")
    elif torch.cuda.is_available() and args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # env setup
    args.env_configs = json.loads(args.env_configs) if args.env_configs else {}
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.env_configs, args.gamma) for i in range(args.num_envs)],
    )
    if args.num_groups > 0:
        env_groups = [list(range(args.num_envs))[i::args.num_groups] for i in range(args.num_groups)]
        print(f"Env Groups: {env_groups}")

    agent = Agent(envs).to(device)
    dtype = next(agent.parameters()).dtype
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Baseline helpers
    # EMA state for baseline_type=ema (Adam-style bias correction)
    ema_m = 0.0
    ema_t = 0
    # Uniform baseline bounds
    if args.baseline_uniform_low is None or args.baseline_uniform_high is None:
        try:
            rlow, rhigh = envs.envs[0].reward_range
            if np.isfinite(rlow) and np.isfinite(rhigh):
                if args.baseline_uniform_low is None:
                    args.baseline_uniform_low = float(rlow)
                if args.baseline_uniform_high is None:
                    args.baseline_uniform_high = float(rhigh)
        except (AttributeError, TypeError, ValueError) as e:
            print(f"Warning: Could not get reward_range from environment: {e}")
            print("Using default values for baseline_uniform bounds")
    if args.baseline_uniform_low is None:
        args.baseline_uniform_low = -1.0
    if args.baseline_uniform_high is None:
        args.baseline_uniform_high = 1.0

    # ALGO Logic: Storage setup (only used when num_steps>0)
    if args.num_steps > 0:
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_checkpoint = 0
    next_log = 0
    return_deque = collections.deque(maxlen=args.num_envs)
    length_deque = collections.deque(maxlen=args.num_envs)

    print(f"Starting training for {args.num_iterations} iterations")
    print(agent)
    print(f"Device: {next(agent.parameters()).device}")

    if args.checkpoint_load_path:
        prev_global_step, prev_iterations = load_checkpoint(
            args.checkpoint_load_path,
            args.checkpoint_param_filters,
            device,
            agent=agent,
            optimizer=optimizer,
        )
        print(f"Loaded checkpoint at step {prev_global_step}, iteration {prev_iterations}")
        writer.add_text(
            "resume",
            f"|param|value|\n|-|-|\n|global_step|{prev_global_step}|\n|iteration|{prev_iterations}|",
        )

    pbar = tqdm(range(args.total_timesteps), desc="Global Steps", dynamic_ncols=True)
    for iteration in range(1, args.num_iterations + 1):
        if args.checkpoint_every > 0 and global_step >= next_checkpoint:
            next_checkpoint = global_step + args.checkpoint_every
            save_checkpoint(run_name, global_step, iteration, agent=agent, optimizer=optimizer)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # ===== TRAINING LOOP STAGES =====
        # Stage 1: Data Collection (rollout or episode)
        # Stage 2: Return Calculation (GAE, TD-n, or MC)
        # Stage 3: Baseline & Advantage Calculation (if not GAE)
        # Stage 4: Policy & Value Network Updates
        
        # ===== STAGE 1: DATA COLLECTION =====
        t_collection_start = time.perf_counter()
        if args.num_steps > 0:
            # Fixed rollout data collection
            (obs, actions, logprobs, rewards, values, next_obs, next_done, global_step) = collect_fixed_rollout(
                agent, writer, dtype, device, envs, obs, dones, values, actions, logprobs, next_obs, next_done, rewards, return_deque, length_deque, global_step, args
            )

            # Flatten the batch for fixed rollouts
            (b_obs, b_logprobs, b_actions, b_values, b_rewards, per_env_rewards_list) = flatten_rollout_batch(
                obs, logprobs, actions, values, rewards, envs, args
            )
        else:
            # Episode mode: collect one full episode from each env (no bootstrapping)
            if args.baseline_type == "same_seed_mean":
                group_seeds = [random.randint(0, 2**31 - 1) for _ in range(args.num_groups)]
                for group_idx, env_idxs in enumerate(env_groups):
                    for env_idx in env_idxs:
                        _obs, _ = envs.envs[env_idx].reset(seed=group_seeds[group_idx])
                        next_obs[env_idx] = torch.tensor(_obs).to(device)
            else:
                next_obs, _ = envs.reset()
                next_obs = torch.tensor(next_obs).to(device)

            # We collect per env first so indexes match between obs, actions, and returns across episodes
            (b_obs, b_actions, b_logprobs, b_values, per_env_rewards_list, global_step, next_obs, per_env_values_list) = collect_episode_rollout(
                agent, writer, dtype, device, envs, return_deque, length_deque, next_obs, global_step, args
            )

        t_collection_end = time.perf_counter()
        
        # ===== STAGE 2: RETURN CALCULATION =====
        t_preprocessing_start = time.perf_counter()
        b_returns_list = []
        per_env_returns = []

        if args.return_type == "gae":
            # GAE returns and advantages
            b_returns, b_advantages, per_env_returns = calculate_gae_returns(
                agent,
                next_obs,
                next_done,
                rewards,
                values,
                dones,
                args,
                per_env_returns,
                b_returns_list,
                device
            )

        elif args.return_type == "td":
            # TD-n where n = num_steps
            # This computes n-step returns with bootstrapping from the value function
            b_returns, b_advantages, per_env_returns = calculate_td_returns(
                agent,
                next_obs,
                next_done,
                rewards,
                dones,
                args,
                values,
                per_env_returns,
                b_returns_list,
                dtype,
                device
            )
        else:  # Monte Carlo Returns
            (b_returns, per_env_returns, b_returns_list) = calculate_mc_returns(
                per_env_rewards_list, args, per_env_returns, b_returns_list, dtype, device
            )
        # ===== STAGE 3: BASELINE CALCULATION (UNIFIED) =====
        if args.return_type == "gae":
            # GAE already computed advantages above
            baseline_vals = b_values
            pass
        elif args.baseline_type == "value" and args.use_value_fn:
            baseline_vals = b_values
            b_advantages = b_returns - baseline_vals
        elif args.baseline_type == "constant":
            baseline_vals = torch.full_like(b_returns, float(args.baseline_constant))
            b_advantages = b_returns - baseline_vals
        elif args.baseline_type == "uniform":
            sampled = random.uniform(float(args.baseline_uniform_low), float(args.baseline_uniform_high))
            baseline_vals = torch.full_like(b_returns, float(sampled))
            b_advantages = b_returns - baseline_vals
        elif args.baseline_type == "stats":
            mean_ret = float(np.mean(b_returns_list))
            std_ret = float(np.std(b_returns_list))
            baseline_vals = torch.normal(
                mean=torch.full_like(b_returns, mean_ret),
                std=torch.full_like(b_returns, std_ret)
            )
            b_advantages = b_returns - baseline_vals
        elif args.baseline_type == "batch_mean":
            mean_ret = float(np.mean(b_returns_list))
            baseline_vals = torch.full_like(b_returns, float(mean_ret))
            b_advantages = b_returns - baseline_vals
        elif args.baseline_type == "ema":
            batch_mean = float(np.mean(b_returns_list))
            ema_t += 1
            beta = float(args.baseline_ema_beta)
            ema_m = beta * ema_m + (1.0 - beta) * batch_mean
            corrected = ema_m / (1.0 - (beta ** ema_t))
            baseline_vals = torch.full_like(b_returns, float(corrected))
            b_advantages = b_returns - baseline_vals
        elif args.baseline_type == "same_seed_mean":
            # Calculate mean over steps for each env in a group, then mean over the group
            env_baseline = {}
            for group_idx, env_idxs in enumerate(env_groups):
                env_means = [np.mean(per_env_returns[env_idx]) for env_idx in env_idxs]
                group_baseline = float(np.mean(env_means))
                for env_idx in env_idxs:
                    env_baseline[env_idx] = group_baseline
            
            baseline_vals = []
            for env_idx in range(args.num_envs):
                baseline_vals.extend([env_baseline[env_idx]] * len(per_env_returns[env_idx]))
            
            baseline_vals = torch.tensor(baseline_vals, dtype=dtype, device=device)
            b_advantages = b_returns - baseline_vals
        else:  # no baseline (REINFORCE with clip & ratio)
            b_advantages = b_returns

        # Final batch setup
        batch_size = b_obs.shape[0]
        minibatch_size = int(max(1, batch_size // args.num_minibatches))
        
        # # Data validation: Check for NaN/Inf values that indicate bugs
        # assert torch.isfinite(b_obs).all(), "NaN or Inf detected in observations"
        # assert torch.isfinite(b_actions).all(), "NaN or Inf detected in actions"
        # assert torch.isfinite(b_logprobs).all(), "NaN or Inf detected in log probabilities"
        # assert torch.isfinite(b_returns).all(), "NaN or Inf detected in returns"
        # assert torch.isfinite(b_advantages).all(), "NaN or Inf detected in advantages"
        # if args.use_value_fn:
        #     assert torch.isfinite(b_values).all(), "NaN or Inf detected in values"
        
        # # Data validation: Check shapes are consistent
        # assert b_obs.shape[0] == batch_size, f"Observation batch size mismatch: {b_obs.shape[0]} != {batch_size}"
        # assert b_actions.shape[0] == batch_size, f"Action batch size mismatch: {b_actions.shape[0]} != {batch_size}"
        # assert b_logprobs.shape[0] == batch_size, f"Logprob batch size mismatch: {b_logprobs.shape[0]} != {batch_size}"
        # assert b_returns.shape[0] == batch_size, f"Returns batch size mismatch: {b_returns.shape[0]} != {batch_size}"
        # assert b_advantages.shape[0] == batch_size, f"Advantages batch size mismatch: {b_advantages.shape[0]} != {batch_size}"

        if args.scale_adv_batch:
            b_advantages /= (b_advantages.std() + 1e-8)

        t_preprocessing_end = time.perf_counter()

        # ===== STAGE 4: TRAINING =====
        t_update_start = time.perf_counter()
        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        grad_norms = []
        policy_ratios = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                (loss, v_loss, pg_loss, entropy_loss, approx_kl, old_approx_kl, clipfracs) = training_minibatch(
                    policy_ratios=policy_ratios,
                    b_advantages=b_advantages,
                    start=start,
                    minibatch_size=minibatch_size,
                    b_inds=b_inds,
                    clipfracs=clipfracs,
                    agent=agent,
                    b_obs=b_obs,
                    b_actions=b_actions,
                    b_logprobs=b_logprobs,
                    b_returns=b_returns if args.use_value_fn else None,
                    b_values=b_values if args.use_value_fn else None,
                    dtype=dtype,
                    device=device,
                    args=args
                )

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                grad_norms.append(grad_norm.item())
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        t_update_end = time.perf_counter()

        # Calculate timing metrics
        time_collection = t_collection_end - t_collection_start
        time_preprocessing = t_preprocessing_end - t_preprocessing_start
        time_update = t_update_end - t_update_start

        # Explained variance (only meaningful when using value baseline)
        if args.use_value_fn and (args.num_steps > 0 or args.baseline_type == "value"):
            y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        else:
            explained_var = np.nan

        log_training_metrics(
            writer=writer,
            optimizer=optimizer,
            iteration=iteration,
            global_step=global_step,
            start_time=start_time,
            next_log=next_log,
            v_loss=v_loss,
            pg_loss=pg_loss,
            entropy_loss=entropy_loss,
            old_approx_kl=old_approx_kl,
            approx_kl=approx_kl,
            clipfracs=clipfracs,
            explained_var=explained_var,
            time_collection=time_collection,
            time_preprocessing=time_preprocessing,
            time_update=time_update,
            rewards=rewards if args.num_steps > 0 else None,
            per_env_rewards_list=per_env_rewards_list,
            b_advantages=b_advantages,
            b_returns=b_returns,
            b_logprobs=b_logprobs,
            b_values=b_values,
            baseline_vals=baseline_vals,
            grad_norms=grad_norms,
            policy_ratios=policy_ratios,
            args=args
        )


        mean_return = sum(return_deque) / len(return_deque) if len(return_deque) > 0 else 0
        pbar.set_postfix_str(f"updates={iteration}, mean_return={mean_return}")
        pbar.update(len(b_obs))

        # Early stop on total timesteps
        if global_step >= args.total_timesteps:
            break

    save_checkpoint(run_name, global_step, iteration, agent=agent, optimizer=optimizer)
    envs.close()
    writer.close()
