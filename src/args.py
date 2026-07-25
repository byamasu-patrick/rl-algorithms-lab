"""
Arguments for configuring the training experiment
"""

import argparse
import os
from argparse import ArgumentParser

def add_args(parser: ArgumentParser):
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__)[: -len(".py")],
                        help="the name of this experiment")
    parser.add_argument("--run-name", type=str, default=None,
                        help="the name of this run")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="grpo2",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="byamasupatrick-rexplore-research-labs",
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--num-checkpoints", type=int, default=10,
                        help="the number of checkpoints to save, set to 0 to disable")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="the number of env steps between checkpoints, set to 0 to disable")
    parser.add_argument("--checkpoint-load-path", type=str, default=None,
                        help="the path to the checkpoint to load")
    parser.add_argument("--log-every", type=int, default=0,
                        help="the number of env steps between detailed stat logging, set to 0 to log every iteration")
    parser.add_argument("--checkpoint-param-filters", type=str, default=None,
                        help="the filter to load checkpoint parameters")
    parser.add_argument("-o", "--overwrite", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="whether to overwrite the run directory if it already exists")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
                        help="the id of the environment")
    parser.add_argument("--env-configs", type=str, default=None,
                        help="the configs of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1100000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=1,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Toggles advantages normalization within the minibatch")
    parser.add_argument("--clip-coef", type=float, default=0.2,
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
    parser.add_argument("--use-value-fn", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Toggle using value function and its loss for baseline and training")
    parser.add_argument("--return-type", type=str, default="gae",
                        choices=["gae", "td", "mc"],
                        help="How to compute returns/advantages: gae (GAE with num_steps), td (n-step TD where n=num_steps), or mc (Monte Carlo with num_steps=0)")
    parser.add_argument("--baseline-type", type=str, default="value",
                        choices=["value", "constant", "uniform", "stats", "batch_mean", "ema", "same_seed_mean"],
                        help="Baseline used to center advantages (ignored when --return-type=gae)")
    parser.add_argument("--scale-adv-batch", type=bool, default=False, action=argparse.BooleanOptionalAction,
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
                        help="number of groups to use")

    # to be filled in runtime
    parser.add_argument("--batch-size", type=int, default=0,
                        help="the batch size (computed in runtime)")
    parser.add_argument("--minibatch-size", type=int, default=0,
                        help="the mini-batch size (computed in runtime)")
    parser.add_argument("--num-iterations", type=int, default=0,
                        help="the number of iterations (computed in runtime)")
