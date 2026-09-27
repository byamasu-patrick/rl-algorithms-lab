"""Returns and advantages for GRPO, as pure functions over trajectories.

A trajectory is one episode (or, with fixed-length rollouts, one episode segment). Every function
takes and returns one float array per trajectory, so the estimators can be tested in isolation.

Group estimators follow DeepSeekMath (Shao et al., 2024, section 4.1):

- outcome supervision: every step of trajectory i gets (R_i - mean(R)) / std(R), where R holds the
  total rewards of the trajectories in i's group;
- process supervision: rewards are normalized by the mean and std of all rewards in the group, and
  step t gets the sum of normalized rewards from t onward.

The remaining estimators are the critic-free return/baseline variants of revisiting-grpo: Monte Carlo
returns minus a baseline computed from the returns themselves. Nothing here uses a value function.
"""

import numpy as np

EPS = 1e-8


def _std(values):
    # Unbiased std, as in reference GRPO implementations; a single sample has no spread.
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def group_members(groups):
    """Map group id -> list of trajectory indices."""
    members = {}
    for idx, group in enumerate(groups):
        members.setdefault(group, []).append(idx)
    return members


def outcome_advantages(rewards, groups):
    """Outcome supervision: each step's advantage is its trajectory's group-normalized total reward."""
    totals = np.array([float(np.sum(r)) for r in rewards])
    advantages = [None] * len(rewards)
    for idxs in group_members(groups).values():
        mean, std = totals[idxs].mean(), _std(totals[idxs])
        for i in idxs:
            advantages[i] = np.full(len(rewards[i]), (totals[i] - mean) / (std + EPS), dtype=np.float32)
    return advantages


def process_advantages(rewards, groups):
    """Process supervision: group-normalize every reward, then sum the normalized rewards to go."""
    advantages = [None] * len(rewards)
    for idxs in group_members(groups).values():
        pooled = np.concatenate([rewards[i] for i in idxs])
        mean, std = pooled.mean(), _std(pooled)
        for i in idxs:
            normalized = (rewards[i] - mean) / (std + EPS)
            advantages[i] = np.cumsum(normalized[::-1])[::-1].astype(np.float32)
    return advantages


def discounted_returns(rewards, gamma, bootstrap=0.0):
    """Discounted reward-to-go, bootstrapped with `bootstrap` after the last step (0 for a terminal)."""
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = bootstrap
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


class BaselineState:
    """Running state for baselines that carry across iterations (the EMA)."""

    def __init__(self):
        self.ema_m = 0.0
        self.ema_t = 0


def baseline_values(baseline_type, returns, groups, args, state, rng):
    """Per-step, critic-free baselines for the revisiting-grpo variants; `returns` is per trajectory."""
    flat = np.concatenate(returns)
    if baseline_type == "constant":
        scalar = float(args.baseline_constant)
    elif baseline_type == "uniform":
        scalar = float(rng.uniform(args.baseline_uniform_low, args.baseline_uniform_high))
    elif baseline_type == "stats":
        # A random baseline drawn per step from the batch return statistics.
        return [rng.normal(flat.mean(), flat.std(), size=len(r)).astype(np.float32) for r in returns]
    elif baseline_type == "batch_mean":
        scalar = float(flat.mean())
    elif baseline_type == "ema":
        state.ema_t += 1
        beta = float(args.baseline_ema_beta)
        state.ema_m = beta * state.ema_m + (1.0 - beta) * float(flat.mean())
        scalar = state.ema_m / (1.0 - beta ** state.ema_t)
    elif baseline_type == "same_seed_mean":
        # Mean over steps for each trajectory, then mean over the trajectories of the group.
        baselines = [None] * len(returns)
        for idxs in group_members(groups).values():
            group_mean = float(np.mean([returns[i].mean() for i in idxs]))
            for i in idxs:
                baselines[i] = np.full(len(returns[i]), group_mean, dtype=np.float32)
        return baselines
    else:
        raise ValueError(f"unknown baseline type {baseline_type!r}")
    return [np.full(len(r), scalar, dtype=np.float32) for r in returns]


def sequence_weights(lengths, aggregation):
    """Per-step loss weights: 1/|o_i| for the paper's per-sequence mean, 1 for a per-token mean."""
    if aggregation == "sequence":
        return [np.full(n, 1.0 / n, dtype=np.float32) for n in lengths]
    return [np.ones(n, dtype=np.float32) for n in lengths]
