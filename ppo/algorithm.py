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
from src.agent import Agent

def parse_args():
    parser = argparse.ArgumentParser()

    filename = os.path.basename(__file__).rstrip(".py")
    exp_name = filename if filename == "ppo" else "ppo"

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

    agent = Agent(envs).to(device)

    print(agent)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Algorithm Logic: Storage setup
    observations = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()

    next_obs_np, info = envs.reset(seed=args.seed)
    next_obs = torch.from_numpy(next_obs_np).float().to(device)
    next_done = torch.zeros(args.num_envs).float().to(device)
    num_updates = args.total_timesteps // args.batch_size

    # Learning loop annealing

    for update in range(1, num_updates + 1):
        if args.annealing_lr:
            # decrease learning rate linearly
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs

            observations[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_np, reward_np, terminated_np, truncated_np, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward_np).to(device).view(-1)
            next_obs = torch.from_numpy(next_obs_np).float().to(device)
            next_done = torch.tensor(np.logical_or(terminated_np, truncated_np)).float().to(device).view(-1)

            if "episode" in info:
                  for idx, d in enumerate(info["_episode"]):
                    if d:
                        episodic_return = info["episode"]["r"][idx]
                        episodic_length = info["episode"]["l"][idx]
                        print(f"global_step={global_step}, episodic_return={episodic_return}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", episodic_length, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return    

                advantages = returns - values        


        batch_observations = observations.reshape((-1,) + envs.single_observation_space.shape)
        batch_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        batch_logprobs = logprobs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values.reshape(-1)

        # Mini-batch update

        b_inds = np.arange(args.batch_size)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(batch_observations[mb_inds], batch_actions.long()[mb_inds])
                logratio = newlogprob - batch_logprobs[mb_inds]
                ratio = logratio.exp()

                # Stats
                # with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs = [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = batch_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - batch_returns[mb_inds]) ** 2
                    v_clipped = batch_values[mb_inds] + torch.clamp(
                        newvalue - batch_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - batch_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - batch_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = batch_values.cpu().numpy(), batch_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


        # Early stopping and logging
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", clipfracs[0], global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        print ("SPS:", int(global_step / (time.time() - start_time)))

        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()