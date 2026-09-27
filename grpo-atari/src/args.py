"""
Arguments for configuring the training experiment.

The flag set follows revisiting-grpo: standard PPO flags, plus return/baseline controls that turn
PPO into critic-free GRPO variants, plus Atari preprocessing flags.
"""

import argparse
import os
import sys
from argparse import ArgumentParser


def add_args(parser: ArgumentParser):
    parser.add_argument("--exp-name", type=str, default=os.path.splitext(os.path.basename(sys.argv[0]))[0],
                        help="the name of this experiment")
    parser.add_argument("--run-name", type=str, default=None,
                        help="the name of this run; defaults to {exp_name}__{env_id}__{seed}")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="grpo-atari",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--num-checkpoints", type=int, default=10,
                        help="the number of checkpoints to save, set to 0 to disable")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="the number of env steps between checkpoints, set to 0 to disable")
    parser.add_argument("--checkpoint-load-path", type=str, default=None,
                        help="the path to the checkpoint to load")
    parser.add_argument("--checkpoint-param-filters", type=str, default=None,
                        help="JSON filter selecting which checkpoint parameters to load")
    parser.add_argument("--log-every", type=int, default=0,
                        help="the number of env steps between detailed stat logging, set to 0 to log every iteration")
    parser.add_argument("-o", "--overwrite", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="whether to overwrite the run directory if it already exists")
    parser.add_argument("--save-model", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="whether to save the final model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
                        help="the user or org name of the model repository from the Hugging Face Hub")

    # Environment arguments
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
                        help="the id of the environment")
    parser.add_argument("--env-configs", type=str, default=None,
                        help="JSON dict of extra keyword arguments passed to gym.make")
    parser.add_argument("--noop-max", type=int, default=30,
                        help="maximum number of random no-ops at the start of an episode, 0 to disable")
    parser.add_argument("--frame-skip", type=int, default=4,
                        help="the number of frames each action is repeated for (max-pooled over the last two)")
    parser.add_argument("--frame-stack", type=int, default=4,
                        help="the number of most recent frames stacked as the observation")
    parser.add_argument("--screen-size", type=int, default=84,
                        help="the height and width observations are resized to")
    parser.add_argument("--episodic-life", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Toggle ending the training episode when a life is lost")
    parser.add_argument("--clip-rewards", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Toggle clipping rewards to their sign {-1, 0, +1}")
    parser.add_argument("--max-episode-steps", type=int, default=27000,
                        help="truncate episodes after this many agent steps (27000 x frame skip 4 = 108k frames), 0 to disable")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=10000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=0,
                        help="the number of steps to run in each environment per policy rollout, 0 to collect full episodes")
    parser.add_argument("--anneal-lr", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Toggles advantages normalization within the minibatch")
    parser.add_argument("--clip-coef", type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")

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

    # to be filled in runtime
    parser.add_argument("--batch-size", type=int, default=0,
                        help="the batch size (computed in runtime)")
    parser.add_argument("--minibatch-size", type=int, default=0,
                        help="the mini-batch size (computed in runtime)")
    parser.add_argument("--num-iterations", type=int, default=0,
                        help="the number of iterations (computed in runtime)")


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
    if args.upload_model:
        assert args.save_model, "--upload-model requires --save-model."

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

    if args.num_checkpoints > 0:
        args.checkpoint_every = args.total_timesteps // args.num_checkpoints
    return args


def parse_args():
    parser = ArgumentParser()
    add_args(parser)
    return validate_args(parser.parse_args())
