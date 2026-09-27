"""GRPO: critic-free PPO with group-relative advantages, with the return/baseline variants of revisiting-grpo.

The flags and their validation are in place; the training loop is not implemented yet.
"""

import argparse

from src.algorithms.ppo import add_policy_gradient_args

DESCRIPTION = "Group Relative Policy Optimization: PPO without a critic, advantages relative to a group baseline"
DISPLAY_NAME = "GRPO"
IMPLEMENTED = False


def add_args(parser):
    # GRPO defaults: full-episode rollouts, and batch-level scaling instead of per-minibatch normalization.
    add_policy_gradient_args(parser, num_steps=0, norm_adv=False)

    # Return and baseline control
    parser.add_argument("--use-value-fn", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Toggle using value function and its loss for baseline and training")
    parser.add_argument("--return-type", type=str, default="mc",
                        choices=["gae", "td", "mc"],
                        help="How to compute returns/advantages: gae (GAE with num_steps), td (n-step TD where n=num_steps), or mc (Monte Carlo with num_steps=0)")
    parser.add_argument("--baseline-type", type=str, default="batch_mean",
                        choices=["value", "constant", "uniform", "stats", "batch_mean", "ema", "same_seed_mean"],
                        help="Baseline used to center advantages (ignored when --return-type=gae)")
    parser.add_argument("--scale-adv-batch", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Toggle using batch std for advantage scaling")
    parser.add_argument("--baseline-constant", type=float, default=None,
                        help="Constant baseline when --baseline-type=constant")
    parser.add_argument("--baseline-uniform-low", type=float, default=None,
                        help="Low bound for --baseline-type=uniform; if unset tries env.reward_range or falls back to -1")
    parser.add_argument("--baseline-uniform-high", type=float, default=None,
                        help="High bound for --baseline-type=uniform; if unset tries env.reward_range or falls back to 1")
    parser.add_argument("--baseline-ema-beta", type=float, default=0.9,
                        help="EMA beta for --baseline-type=ema (Adam-style bias correction)")
    parser.add_argument("--num-groups", type=int, default=0,
                        help="number of groups of environments sharing a reset seed, for --baseline-type=same_seed_mean")


def validate_args(args):
    """Check flag combinations and fill in the runtime-computed sizes."""
    if args.return_type in ("gae", "td"):
        assert args.num_steps > 0, \
            f"--return-type {args.return_type} requires --num-steps > 0."
    elif args.num_steps != 0:
        print("WARNING: Using num_steps != 0 with --return-type mc. Monte Carlo requires num_steps = 0 (full episodes) for correctness.")

    if args.baseline_type == "same_seed_mean":
        assert args.num_groups > 0, \
            "same_seed_mean baseline requires --num-groups > 0 to define environment groups."
        assert args.num_envs % args.num_groups == 0, \
            f"num_envs ({args.num_envs}) must be divisible by num_groups ({args.num_groups}) for same_seed_mean baseline."
    if args.baseline_type == "constant":
        assert args.baseline_constant is not None, \
            "constant baseline requires --baseline-constant to be set."
    if (args.return_type == "gae" or args.baseline_type == "value") and not args.use_value_fn:
        print(f"WARNING: --return-type {args.return_type} / --baseline-type {args.baseline_type} needs a value function, "
              "but --use-value-fn is off. The value function won't be trained.")

    if args.num_steps > 0:
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(max(1, args.batch_size // args.num_minibatches))
        args.num_iterations = max(1, args.total_timesteps // args.batch_size)
        assert args.batch_size >= args.num_minibatches, \
            f"batch_size ({args.batch_size}) must be >= num_minibatches ({args.num_minibatches})."
    else:
        # Episode mode: batch size varies per iteration; training stops on total timesteps.
        args.batch_size = 0
        args.minibatch_size = 0
        args.num_iterations = max(1, args.total_timesteps)
    return args


def train(args, run, device):
    raise NotImplementedError("GRPO training is not implemented yet.")
