"""DQN (Mnih et al., 2015), ported from dqn-atari onto the shared environment, run and evaluation code."""

import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.environment import log_episodes, make_vector_env
from src.replay_buffer import ReplayBuffer

DESCRIPTION = "Deep Q-Network: replay buffer, target network, epsilon-greedy exploration"
DISPLAY_NAME = "DQN"


def add_args(parser):
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=1000000,
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
                        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
                        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=80000,
                        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
                        help="the frequency of training")


def validate_args(args):
    assert args.learning_starts < args.total_timesteps, "--learning-starts must be below --total-timesteps."
    return args


class QNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, envs.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def epsilon_greedy(q_network, action_space, obs, epsilon, device):
    if random.random() < epsilon:
        return np.array([action_space.sample() for _ in range(len(obs))])
    q_values = q_network(torch.Tensor(obs).to(device))
    return torch.argmax(q_values, dim=1).cpu().numpy()


def train(args, run, device):
    envs = make_vector_env(args, run.name, args.num_envs)

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    run.load(device, q_network=q_network, optimizer=optimizer)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        run.maybe_checkpoint(global_step, global_step, q_network=q_network, optimizer=optimizer)

        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        actions = epsilon_greedy(q_network, envs.single_action_space, obs, epsilon, device)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        log_episodes(run.writer, infos, global_step)

        # Store the true final observation for truncated episodes (autoreset already replaced next_obs).
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    sps = int(global_step / (time.time() - start_time))
                    run.writer.add_scalar("losses/td_loss", loss, global_step)
                    run.writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    run.writer.add_scalar("charts/epsilon", epsilon, global_step)
                    run.writer.add_scalar("charts/SPS", sps, global_step)
                    print("SPS:", sps)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    envs.close()

    def act(obs):
        return epsilon_greedy(q_network, envs.single_action_space, obs, args.end_e, device)

    run.save_evaluate_upload(q_network, act, DISPLAY_NAME)
