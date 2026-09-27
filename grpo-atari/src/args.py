"""
Command-line interface: `python algorithm.py <algo> [flags]`.

Flags shared by every algorithm (experiment setup and the Atari environment) are defined here, so
DQN, PPO and GRPO runs are configured, preprocessed and logged identically. Each algorithm module
in `src/algorithms` adds its own flags to its subcommand.
"""

import argparse
import json
from argparse import ArgumentParser


def add_common_args(parser: ArgumentParser):
    parser.add_argument("--exp-name", type=str, default=None,
                        help="the name of this experiment; defaults to the algorithm name")
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
    parser.add_argument("--wandb-entity", type=str, default="mscsr000324-must",
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--num-checkpoints", type=int, default=10,
                        help="the number of checkpoints to save, set to 0 to disable")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="the number of env steps between checkpoints, set to 0 to disable")
    parser.add_argument("--checkpoint-load-path", type=str, default=None,
                        help="the path to a checkpoint whose weights (and optimizer state) initialize this run")
    parser.add_argument("--checkpoint-param-filters", type=str, default=None,
                        help="JSON dict {object: regex}; only parameters matching the regex are loaded from the checkpoint")
    parser.add_argument("--log-every", type=int, default=0,
                        help="the number of env steps between detailed stat logging, set to 0 to log every iteration")
    parser.add_argument("-o", "--overwrite", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="whether to overwrite the run directory if it already exists")
    parser.add_argument("--save-model", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="whether to save the final model into `runs/{run_name}` and evaluate it")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="the number of evaluation episodes run on the saved model")
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

    # Budget, identical across algorithms so runs are comparable
    parser.add_argument("--total-timesteps", type=int, default=10000000,
                        help="total timesteps of the experiments")


def validate_common_args(args):
    if args.exp_name is None:
        args.exp_name = args.algo
    args.env_configs = json.loads(args.env_configs) if args.env_configs else {}
    args.checkpoint_param_filters = json.loads(args.checkpoint_param_filters) if args.checkpoint_param_filters else {}
    if args.upload_model:
        assert args.save_model, "--upload-model requires --save-model."
    if args.num_checkpoints > 0:
        args.checkpoint_every = args.total_timesteps // args.num_checkpoints
    return args


def parse_args(algorithms):
    """Parse `sys.argv` into the chosen algorithm's namespace; `algorithms` maps name -> module."""
    common = ArgumentParser(add_help=False)
    add_common_args(common)

    parser = ArgumentParser(description="DQN, PPO and GRPO on Atari with shared preprocessing and logging.")
    subparsers = parser.add_subparsers(dest="algo", required=True, metavar="{" + ",".join(algorithms) + "}")
    for name, module in algorithms.items():
        sub = subparsers.add_parser(name, parents=[common], help=module.DESCRIPTION)
        module.add_args(sub)

    args = parser.parse_args()
    validate_common_args(args)
    algorithms[args.algo].validate_args(args)
    return args
