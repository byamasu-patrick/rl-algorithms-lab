import argparse
import os
import time
import random
import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()

    filename = os.path.basename(__file__).rstrip(".py")
    exp_name = filename if filename == "dqn" else "dqn"

    parser.add_argument("--exp-name", type=str, default=exp_name,
                        help="the name of this experiment")

    parser.add_argument("--gym-id", type=str, default="CartPole-v1", help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="the learning rate of the optimizer")
    
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=25000, help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, 
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, 
                        help="if toggled, cuda will be enabled by default")

    parser.add_argument("--track", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True, 
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")

    parser.add_argument("--capture-video", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")

    parser.add_argument("--num-envs", type=int, default=4, help="the number of parallel game environments")

    parser.add_argument("--num-steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout")

    parser.add_argument("--annealing-lr", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, the learning rate will be annealed")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")

    parser.add_argument("--gae", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage estimation")

    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")

    parser.add_argument("--num-minibatches", type=int, default=4, help="the number of mini-batches")

    parser.add_argument("--update-epochs", type=int, default=4, help="the K epochs to update the policy")

    parser.add_argument("--norm-adv", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")

    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")

    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")

    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")

    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")

    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    
    return args


if __name__ == "__main__":
    args = parse_args()

    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            # monitor_gym=True,
            save_code=True,
        )
    

    writer = SummaryWriter(f"runs/{run_name}")

    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % (
        "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    def make_env(gym_id, seed, idx, capture_video, run_name):

        def thunk():
            env = gym.make(gym_id, render_mode="rgb_array")
            env = gym.wrappers.RecordEpisodeStatistics(env, stats_key="episode")

            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}" if args.track else None, episode_trigger=lambda x: x % 100 == 0)

            # env.env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) 
         for i in range(args.num_envs)]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    print("envs.single_action_space.shape", envs.single_action_space.shape)
    print("envs.single_action_space.n", envs.single_action_space.n)