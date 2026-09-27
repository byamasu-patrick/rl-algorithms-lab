import json
import os
import random
import shutil
import time

import gymnasium as gym
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from hf_jobs import launch
from src.args import parse_args
from src.environment import make_env


if __name__ == "__main__":
    launch()  # with --hf-job, submits this run to Hugging Face Jobs and exits
    args = parse_args()
    args.env_configs = json.loads(args.env_configs) if args.env_configs else {}
    args.checkpoint_param_filters = json.loads(args.checkpoint_param_filters) if args.checkpoint_param_filters else {}

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{args.exp_name}__{args.env_id.replace('/', '_').replace('-', '_').lower()}__{args.seed}"

    if os.path.exists(f"runs/{run_name}/config.yaml"):
        if args.overwrite:
            print(f"Run directory {run_name} already exists. Overwriting.")
            shutil.rmtree(f"runs/{run_name}")
        else:
            print(f"Run directory {run_name} already exists. Exiting.")
            exit(0)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            group=args.exp_name,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % (
        "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    with open(f"runs/{run_name}/config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(vars(args), f)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args)
         for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    print("envs.single_observation_space.shape", envs.single_observation_space.shape)
    print("envs.single_action_space.n", envs.single_action_space.n)

    # TODO: agent, rollout collection, returns and baselines, and the GRPO update.

    envs.close()
    writer.close()
