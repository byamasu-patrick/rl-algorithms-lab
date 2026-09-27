"""GRPO (Shao et al., 2024): PPO's clipped objective with the critic removed.

There is no value function and no GAE. The baseline comes from a *group* of episodes started from the
same state, and the update maximizes the GRPO objective (DeepSeekMath, eq. 3):

    J(θ) = E[ 1/G Σ_i 1/|o_i| Σ_t ( min(ρ_{i,t} Â_{i,t}, clip(ρ_{i,t}, 1-ε, 1+ε) Â_{i,t}) - β D_KL[π_θ || π_ref] ) ]

with ρ = π_θ / π_θ_old, and D_KL estimated per step with the unbiased estimator
π_ref/π_θ - log(π_ref/π_θ) - 1. Mapped onto Atari:

- a "question" q is a start state, fixed by a reset seed;
- the group {o_1..o_G} is G environments reset with the same seed, so they start identically and
  diverge only through the policy's sampling (NoFrameskip-v4 has no sticky actions);
- an "output" o_i is one episode, and its reward r_i the episode's total (clipped) reward.

`--advantage-type` selects outcome supervision (section 4.1.2, the default), process supervision
(4.1.3), or a critic-free baseline from revisiting-grpo (Monte Carlo returns minus a batch, group,
EMA, random, or constant baseline).
"""

import argparse
import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.advantages import (
    BaselineState,
    baseline_values,
    discounted_returns,
    group_members,
    outcome_advantages,
    process_advantages,
    sequence_weights,
)
from src.algorithms.ppo import add_policy_gradient_args, layer_init
from src.environment import log_episodes, make_vector_env

DESCRIPTION = "Group Relative Policy Optimization: no critic, advantages relative to a group of episodes"
DISPLAY_NAME = "GRPO"


def add_args(parser):
    add_policy_gradient_args(parser, num_steps=0, norm_adv=False, num_envs=16)
    # A seeded group reset must restart the game; with life-loss episodes it would not (see validate_args).
    # Whole games as episodes also follow Machado et al. (2018), who recommend not using life-loss signals.
    parser.set_defaults(episodic_life=False)

    # GRPO objective
    parser.add_argument("--advantage-type", type=str, default="outcome",
                        choices=["outcome", "process", "baseline"],
                        help="outcome: group-normalized episode reward on every step (DeepSeekMath 4.1.2); "
                             "process: sum of group-normalized rewards to go (4.1.3); "
                             "baseline: Monte Carlo returns minus --baseline-type (revisiting-grpo)")
    parser.add_argument("--num-groups", type=int, default=2,
                        help="groups of environments sharing a reset seed per iteration; group size is "
                             "num_envs / num_groups; 0 treats the whole batch as one unseeded group")
    parser.add_argument("--kl-coef", type=float, default=0.04,
                        help="beta: weight of the KL penalty to the reference policy (0 disables it)")
    parser.add_argument("--ref-update-every", type=int, default=10,
                        help="iterations between copies of the policy into the reference policy; 0 keeps the initial policy")
    parser.add_argument("--loss-aggregation", type=str, default="sequence", choices=["sequence", "token"],
                        help="sequence: mean over each trajectory, then over trajectories (1/G sum 1/|o_i| sum_t); "
                             "token: mean over all steps")

    # Critic-free baselines, for --advantage-type baseline
    parser.add_argument("--baseline-type", type=str, default="batch_mean",
                        choices=["constant", "uniform", "stats", "batch_mean", "ema", "same_seed_mean"],
                        help="baseline subtracted from the Monte Carlo returns with --advantage-type baseline")
    parser.add_argument("--scale-adv-batch", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Toggle dividing --advantage-type baseline advantages by the batch std")
    parser.add_argument("--baseline-constant", type=float, default=None,
                        help="Constant baseline when --baseline-type=constant")
    parser.add_argument("--baseline-uniform-low", type=float, default=None,
                        help="Low bound for --baseline-type=uniform; if unset tries env.reward_range or falls back to -1")
    parser.add_argument("--baseline-uniform-high", type=float, default=None,
                        help="High bound for --baseline-type=uniform; if unset tries env.reward_range or falls back to 1")
    parser.add_argument("--baseline-ema-beta", type=float, default=0.9,
                        help="EMA beta for --baseline-type=ema (Adam-style bias correction)")


def validate_args(args):
    """Check flag combinations and fill in the runtime-computed sizes."""
    episode_mode = args.num_steps == 0

    if args.advantage_type in ("outcome", "process"):
        assert episode_mode, \
            f"--advantage-type {args.advantage_type} scores whole episodes; it requires --num-steps 0."
    else:
        if args.baseline_type == "same_seed_mean":
            assert args.num_groups > 0, "--baseline-type same_seed_mean requires --num-groups > 0."
        if args.baseline_type == "constant":
            assert args.baseline_constant is not None, "--baseline-type constant requires --baseline-constant."
        if not episode_mode:
            print("WARNING: --num-steps > 0 gives Monte Carlo returns over truncated episode segments with no "
                  "bootstrap (GRPO has no critic), as in revisiting-grpo's fixed-horizon runs.")

    if args.num_groups > 0:
        assert args.num_envs % args.num_groups == 0, \
            f"--num-envs ({args.num_envs}) must be divisible by --num-groups ({args.num_groups})."
        if args.advantage_type in ("outcome", "process"):
            assert args.num_envs // args.num_groups >= 2, \
                "each group needs at least 2 environments; a single member has no spread, so every advantage is 0."
        if episode_mode:
            assert not args.episodic_life, \
                "seeded groups need whole-game episodes: resetting mid-game under --episodic-life does not restart " \
                "the game. Use --no-episodic-life (the GRPO default)."
    assert args.kl_coef >= 0, "--kl-coef must be >= 0."

    if episode_mode:
        # Episode lengths vary, so batch sizes are set per iteration; training stops on total timesteps.
        args.batch_size = 0
        args.minibatch_size = 0
        args.num_iterations = 0
    else:
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(max(1, args.batch_size // args.num_minibatches))
        args.num_iterations = max(1, args.total_timesteps // args.batch_size)
        assert args.batch_size >= args.num_minibatches, \
            f"batch_size ({args.batch_size}) must be >= num_minibatches ({args.num_minibatches})."
    return args


class Policy(nn.Module):
    """PPO's Atari actor without the critic head: Nature CNN trunk and action logits only."""

    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)

    def get_action(self, x, action=None):
        """Sample (or score) actions; returns (action, log-probability, entropy)."""
        probs = Categorical(logits=self.actor(self.network(x / 255.0)))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()


@dataclass
class Trajectory:
    """One episode (or, with --num-steps > 0, one episode segment) from one environment."""
    group: int
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)

    def __len__(self):
        return len(self.rewards)


def env_groups(args):
    """Group id of each environment: environment i belongs to group i % num_groups."""
    if args.num_groups > 0:
        return [i % args.num_groups for i in range(args.num_envs)]
    return [0] * args.num_envs


def episode_stats(info):
    """(return, length) when `info` reports the end of a whole game, else None."""
    if "episode" not in info:
        return None
    episode = info["episode"]
    return float(np.asarray(episode["r"]).reshape(-1)[0]), int(np.asarray(episode["l"]).reshape(-1)[0])


def sample_actions(policy, obs_list, device):
    obs = torch.as_tensor(np.stack(obs_list), device=device).float()
    with torch.no_grad():
        action, logprob, _ = policy.get_action(obs)
    return action.cpu().numpy(), logprob.cpu().numpy()


def collect_episodes(policy, envs, current_obs, groups, seed_rng, args, device, writer, global_step):
    """Play one episode in every environment; seeded groups restart from a shared start state.

    Environments are stepped individually, so the ones that finish early stop consuming steps while
    the rest of the group plays on. Returns (trajectories, steps taken).
    """
    singles = envs.envs
    if args.num_groups > 0:
        seeds = seed_rng.integers(0, 2**31 - 1, size=args.num_groups)
        for i, env in enumerate(singles):
            obs, _ = env.reset(seed=int(seeds[groups[i]]))
            current_obs[i] = np.asarray(obs)

    trajectories = [Trajectory(group=groups[i]) for i in range(len(singles))]
    active = list(range(len(singles)))
    steps = 0
    while active:
        actions, logprobs = sample_actions(policy, [current_obs[i] for i in active], device)
        still_active = []
        for k, i in enumerate(active):
            next_obs, reward, terminated, truncated, info = singles[i].step(actions[k])
            trajectory = trajectories[i]
            trajectory.obs.append(current_obs[i])
            trajectory.actions.append(actions[k])
            trajectory.logprobs.append(logprobs[k])
            trajectory.rewards.append(float(reward))
            steps += 1

            stats = episode_stats(info)
            if stats is not None:
                print(f"global_step={global_step + steps}, episodic_return={stats[0]}")
                writer.add_scalar("charts/episodic_return", stats[0], global_step + steps)
                writer.add_scalar("charts/episodic_length", stats[1], global_step + steps)

            if terminated or truncated:
                if args.num_groups == 0:
                    # Unseeded: carry on from where this episode leaves the game (next life, or a new game).
                    next_obs, _ = singles[i].reset()
            else:
                still_active.append(i)
            current_obs[i] = np.asarray(next_obs)
        active = still_active
    return trajectories, steps


def collect_fixed_rollout(policy, envs, next_obs, groups, args, device, writer, global_step):
    """`--num-steps` steps in every environment, split into trajectories at episode boundaries."""
    open_segments = [Trajectory(group=groups[i]) for i in range(args.num_envs)]
    trajectories = []
    for step in range(args.num_steps):
        actions, logprobs = sample_actions(policy, list(next_obs), device)
        obs_before = next_obs
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        log_episodes(writer, infos, global_step + (step + 1) * args.num_envs)
        for i in range(args.num_envs):
            segment = open_segments[i]
            segment.obs.append(np.asarray(obs_before[i]))
            segment.actions.append(actions[i])
            segment.logprobs.append(logprobs[i])
            segment.rewards.append(float(rewards[i]))
            if terminations[i] or truncations[i]:
                trajectories.append(segment)
                open_segments[i] = Trajectory(group=groups[i])
    # Segments cut by the rollout boundary are kept as they are: without a critic there is no bootstrap.
    trajectories += [segment for segment in open_segments if len(segment) > 0]
    return trajectories, next_obs


def compute_advantages(trajectories, args, baseline_state, rng):
    """Per-step advantages for each trajectory, from group statistics only (no value function)."""
    rewards = [np.asarray(t.rewards, dtype=np.float32) for t in trajectories]
    groups = [t.group for t in trajectories]

    if args.advantage_type == "outcome":
        return outcome_advantages(rewards, groups)
    if args.advantage_type == "process":
        return process_advantages(rewards, groups)

    returns = [discounted_returns(r, args.gamma) for r in rewards]
    baselines = baseline_values(args.baseline_type, returns, groups, args, baseline_state, rng)
    advantages = [ret - base for ret, base in zip(returns, baselines)]
    if args.scale_adv_batch:
        std = float(np.concatenate(advantages).std())
        advantages = [a / (std + 1e-8) for a in advantages]
    return advantages


def kl_estimate(ref_logprob, logprob):
    """Per-step unbiased estimate of KL(pi_theta || pi_ref): pi_ref/pi_theta - log(pi_ref/pi_theta) - 1 >= 0."""
    log_ratio = ref_logprob - logprob
    return log_ratio.exp() - log_ratio - 1


def grpo_loss(logprob, old_logprob, advantages, entropy, weights, ref_logprob, clip_coef, kl_coef, ent_coef):
    """Negative GRPO objective over a minibatch of steps.

    Per step: -(min(rho A, clip(rho, 1-eps, 1+eps) A) - beta KL) - ent_coef H. Steps are averaged with
    `weights` (1/|o_i| for the paper's 1/G sum_i 1/|o_i| sum_t). Returns (loss, policy loss, mean KL or None).
    """
    ratio = (logprob - old_logprob).exp()
    surrogate = torch.min(ratio * advantages, torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * advantages)
    per_step = -surrogate - ent_coef * entropy
    kl = None
    if ref_logprob is not None:
        kl = kl_estimate(ref_logprob, logprob)
        per_step = per_step + kl_coef * kl
    total_weight = weights.sum()
    loss = (per_step * weights).sum() / total_weight
    pg_loss = (-surrogate * weights).sum() / total_weight
    return loss, pg_loss, None if kl is None else kl.mean()


def degenerate_group_fraction(trajectories):
    """Fraction of groups whose episodes all scored the same, which gives every member advantage 0."""
    totals = [sum(t.rewards) for t in trajectories]
    groups = group_members([t.group for t in trajectories]).values()
    return float(np.mean([np.ptp([totals[i] for i in idxs]) == 0 for idxs in groups]))


def train(args, run, device):
    envs = make_vector_env(args, run.name, args.num_envs)
    policy = Policy(envs).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)
    run.load(device, policy=policy, optimizer=optimizer)
    reference = copy.deepcopy(policy).requires_grad_(False) if args.kl_coef > 0 else None

    # Uniform baseline bounds: the flags, else the env's reward range if finite, else [-1, 1].
    low, high = envs.envs[0].reward_range
    if args.baseline_uniform_low is None:
        args.baseline_uniform_low = float(low) if np.isfinite(low) else -1.0
    if args.baseline_uniform_high is None:
        args.baseline_uniform_high = float(high) if np.isfinite(high) else 1.0

    groups = env_groups(args)
    seed_rng = np.random.default_rng(args.seed)
    baseline_state = BaselineState()
    obs, _ = envs.reset(seed=args.seed)
    current_obs = [np.asarray(o) for o in obs]

    global_step = 0
    iteration = 0
    next_log = 0
    start_time = time.time()
    while global_step < args.total_timesteps:
        iteration += 1
        run.maybe_checkpoint(global_step, iteration, policy=policy, optimizer=optimizer)
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - global_step / args.total_timesteps) * args.learning_rate
        # Algorithm 1: the reference model is reset to the current policy every --ref-update-every iterations.
        if reference is not None and args.ref_update_every > 0 and (iteration - 1) % args.ref_update_every == 0:
            reference.load_state_dict(policy.state_dict())

        # Stage 1: sample a group of episodes per start state from the old policy
        t_collect = time.perf_counter()
        if args.num_steps == 0:
            trajectories, steps = collect_episodes(
                policy, envs, current_obs, groups, seed_rng, args, device, run.writer, global_step)
        else:
            trajectories, current_obs = collect_fixed_rollout(
                policy, envs, current_obs, groups, args, device, run.writer, global_step)
            steps = args.num_envs * args.num_steps
        global_step += steps
        t_collect = time.perf_counter() - t_collect

        # Stage 2: group-relative advantages and the 1/|o_i| loss weights
        advantages = compute_advantages(trajectories, args, baseline_state, seed_rng)
        weights = sequence_weights([len(t) for t in trajectories], args.loss_aggregation)

        b_obs = np.stack([o for t in trajectories for o in t.obs])  # uint8, kept on the CPU
        b_actions = torch.as_tensor(np.concatenate([t.actions for t in trajectories]), device=device).long()
        b_logprobs = torch.as_tensor(np.concatenate([t.logprobs for t in trajectories]), device=device).float()
        b_advantages = torch.as_tensor(np.concatenate(advantages), device=device).float()
        b_weights = torch.as_tensor(np.concatenate(weights), device=device).float()
        batch_size = len(b_obs)
        minibatch_size = max(1, -(-batch_size // args.num_minibatches))

        def obs_batch(idx):
            return torch.as_tensor(b_obs[idx], device=device).float()

        b_ref_logprobs = None
        if reference is not None:
            with torch.no_grad():
                b_ref_logprobs = torch.cat([
                    reference.get_action(obs_batch(np.arange(s, min(s + minibatch_size, batch_size))),
                                         b_actions[s:s + minibatch_size])[1]
                    for s in range(0, batch_size, minibatch_size)
                ])

        # Stage 3: maximize the GRPO objective for mu = --update-epochs passes over the batch
        t_update = time.perf_counter()
        b_inds = np.arange(batch_size)
        clipfracs, kl_refs = [], []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start:start + minibatch_size]
                _, newlogprob, entropy = policy.get_action(obs_batch(mb), b_actions[mb])
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                loss, pg_loss, kl_ref = grpo_loss(
                    newlogprob, b_logprobs[mb], mb_advantages, entropy, b_weights[mb],
                    None if b_ref_logprobs is None else b_ref_logprobs[mb],
                    args.clip_coef, args.kl_coef, args.ent_coef)
                if kl_ref is not None:
                    kl_refs.append(kl_ref.item())

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        t_update = time.perf_counter() - t_update

        # Logging
        sps = int(global_step / (time.time() - start_time))
        print(f"iteration={iteration}, global_step={global_step}, batch={batch_size}, SPS={sps}")
        run.writer.add_scalar("charts/SPS", sps, global_step)
        if global_step >= next_log:
            next_log = global_step + args.log_every
            w = run.writer
            flat_advantages = np.concatenate(advantages)
            w.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            w.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            w.add_scalar("losses/entropy", entropy.mean().item(), global_step)
            w.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            w.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            w.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            if kl_refs:
                w.add_scalar("losses/kl_ref", np.mean(kl_refs), global_step)
            w.add_scalar("grpo/advantage_mean", float(flat_advantages.mean()), global_step)
            w.add_scalar("grpo/advantage_std", float(flat_advantages.std()), global_step)
            w.add_scalar("grpo/degenerate_group_fraction", degenerate_group_fraction(trajectories), global_step)
            w.add_scalar("rollout/batch_size", batch_size, global_step)
            w.add_scalar("rollout/trajectories", len(trajectories), global_step)
            w.add_scalar("rollout/mean_trajectory_length", batch_size / len(trajectories), global_step)
            w.add_scalar("rollout/mean_trajectory_reward", float(np.mean([sum(t.rewards) for t in trajectories])), global_step)
            w.add_scalar("time/collection", t_collect, global_step)
            w.add_scalar("time/update", t_update, global_step)

    envs.close()

    def act(obs):
        with torch.no_grad():
            action, _, _ = policy.get_action(torch.Tensor(np.asarray(obs)).to(device))
        return action.cpu().numpy()

    run.save_evaluate_upload(policy, act, DISPLAY_NAME)
