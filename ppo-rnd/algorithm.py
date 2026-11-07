import argparse
import os
import time
import random
from collections import deque
import numpy as np
import torch
import ale_py
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv,
)
from torch.utils.tensorboard import SummaryWriter
from hf_jobs import launch
from src.agent import Agent
from src.rnd import RNDModel, RunningMeanStd, RewardForwardFilter


def strtobool(value):
    # distutils.util.strtobool, which the sibling projects import, was removed with distutils in Python 3.12
    value = value.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return 1
    if value in ("n", "no", "f", "false", "off", "0"):
        return 0
    raise ValueError(f"invalid truth value {value!r}")


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment setup
    parser.add_argument("--exp-name", type=str, default="ppo_rnd",
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to save the agent and RND predictor into the `runs/{run_name}` folder")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="with --save-model, also checkpoint every this many iterations (0: only at the end)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="ALE/MontezumaRevenge-v5",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=2000000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=128,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--async-envs", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="run the environments in subprocesses (AsyncVectorEnv) instead of in-process")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.999,
                        help="the discount factor gamma (extrinsic)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.001,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    parser.add_argument("--sticky-action-prob", type=float, default=0.25,
                        help="probability that the emulator repeats the previous action (Machado et al., 2018)")

    # RND arguments
    parser.add_argument("--update-proportion", type=float, default=0.25,
                        help="proportion of experience used for the predictor update")
    parser.add_argument("--int-coef", type=float, default=1.0,
                        help="coefficient of the intrinsic advantage")
    parser.add_argument("--ext-coef", type=float, default=2.0,
                        help="coefficient of the extrinsic advantage")
    parser.add_argument("--int-gamma", type=float, default=0.99,
                        help="intrinsic reward discount rate")
    parser.add_argument("--num-iterations-obs-norm-init", type=int, default=50,
                        help="rollouts of random actions used to initialise the observation normalisation")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    return args


def make_env(env_id, seed, idx, capture_video, run_name, sticky_action_prob):
    # The same preprocessing as CleanRL's envpool settings (episodic_life, reward_clip, sticky actions,
    # 4-frame skip with max-pooling, 84x84 grayscale, 4-frame stack, 27,000-step limit), built from
    # Gymnasium and Stable-Baselines3 wrappers because envpool has no Windows build.

    def thunk():
        gym.register_envs(ale_py)
        # frameskip=1 because MaxAndSkipEnv does the skipping; the time limit is in raw frames (27,000 agent steps)
        env = gym.make(
            env_id,
            frameskip=1,
            repeat_action_probability=sticky_action_prob,
            max_episode_steps=108000,
            render_mode="rgb_array" if capture_video and idx == 0 else None,
        )
        # records whole games: it sits inside EpisodicLifeEnv, so a lost life doesn't end the recorded episode
        env = gym.wrappers.RecordEpisodeStatistics(env, stats_key="episode")
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.FrameStackObservation(env, 4)
        env.action_space.seed(seed)
        return env
    return thunk


def last_frame(obs_batch):
    """The newest frame of each 4-frame stack, shaped (N, 1, 84, 84): what the RND networks read."""
    return obs_batch[:, 3:4, :, :]


if __name__ == "__main__":
    launch()  # with --hf-job, submits this run to Hugging Face Jobs and exits
    args = parse_args()

    run_name = f"{args.env_id.replace('/', '-')}__{args.exp_name}__{args.seed}__{int(time.time())}"

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

    # SAME_STEP autoreset: when an episode ends, `next_obs` is already the new episode's first frame, so
    # dones[t] marks obs[t] as an episode start exactly as the classic gym API that CleanRL's PPO assumes.
    vector_env = gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    envs = vector_env(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.sticky_action_prob)
         for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    print("envs.single_observation_space.shape", envs.single_observation_space.shape)
    print("envs.single_action_space.n", envs.single_action_space.n)

    agent = Agent(envs).to(device)
    rnd_model = RNDModel(1, 512).to(device)
    # one optimizer for the policy, both value heads and the RND predictor (the target is frozen)
    combined_parameters = list(agent.parameters()) + list(rnd_model.predictor.parameters())
    optimizer = optim.Adam(combined_parameters, lr=args.learning_rate, eps=1e-5)

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    discounted_reward = RewardForwardFilter(args.int_gamma)

    def save_checkpoint():
        # the run directory is copied back from a Hugging Face Job even when it hits its timeout,
        # so periodic checkpoints survive runs far shorter than --total-timesteps
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(
            {
                "agent": agent.state_dict(),
                "rnd_predictor": rnd_model.predictor.state_dict(),
                "rnd_target": rnd_model.target.state_dict(),
                "obs_rms": {"mean": obs_rms.mean, "var": obs_rms.var, "count": obs_rms.count},
                "global_step": global_step,
            },
            model_path,
        )
        print(f"model saved to {model_path}")

    def normalize_obs(frames):
        # whiten each pixel with the running statistics, then clip, so the fixed target network's
        # output varies with the observation rather than with its overall brightness
        mean = torch.as_tensor(obs_rms.mean, dtype=torch.float32, device=device)
        std = torch.as_tensor(np.sqrt(obs_rms.var), dtype=torch.float32, device=device)
        return ((frames - mean) / std).clip(-5, 5)

    # Algorithm Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    curiosity_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs_np).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Initialise the observation statistics from random play, so the first intrinsic rewards are
    # computed on sensibly normalised frames rather than on raw pixel values.
    print("Start to initialize observation normalization parameter.....")
    next_ob = []
    for step in range(args.num_steps * args.num_iterations_obs_norm_init):
        acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
        s, _, term, trunc, _ = envs.step(acs)
        next_ob.append(last_frame(s))
        if len(next_ob) % args.num_steps == 0:
            obs_rms.update(np.concatenate(next_ob).astype(np.float64))
            next_ob = []
        # training continues from where random play left the games
        next_obs = torch.Tensor(s).to(device)
        next_done = torch.Tensor(np.logical_or(term, trunc)).to(device)
    print("End to initialize...")

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value_ext, value_int = agent.get_action_and_value(obs[step])
                ext_values[step], int_values[step] = value_ext.flatten(), value_int.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(np.logical_or(terminations, truncations)).to(device)

            # The intrinsic reward is the novelty of the state the action led to. Under SAME_STEP
            # autoreset that state is in `final_obs` for environments whose episode just ended.
            real_next_obs = next_obs_np
            if "final_obs" in infos:
                real_next_obs = next_obs_np.copy()
                for idx, final_obs in enumerate(infos["final_obs"]):
                    if final_obs is not None:
                        real_next_obs[idx] = final_obs
            rnd_next_obs = normalize_obs(last_frame(torch.Tensor(real_next_obs).to(device)))
            curiosity_rewards[step] = rnd_model.intrinsic_reward(rnd_next_obs)

            # Episode statistics. SyncVectorEnv flattens the nested `episode` dict into
            # per-key arrays, each with a matching `_key` boolean mask over environments.
            if "final_info" in infos and "episode" in infos["final_info"]:
                episode = infos["final_info"]["episode"]
                for idx, finished in enumerate(infos["final_info"]["_episode"]):
                    if finished:
                        avg_returns.append(episode["r"][idx])
                        print(f"global_step={global_step}, episodic_return={episode['r'][idx]}, "
                              f"curiosity_reward={curiosity_rewards[step].mean().item():.4f}")
                        writer.add_scalar("charts/episodic_return", episode["r"][idx], global_step)
                        writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                        writer.add_scalar("charts/episodic_length", episode["l"][idx], global_step)
                        writer.add_scalar("charts/episode_curiosity_reward", curiosity_rewards[step][idx].item(), global_step)

        # Normalise the intrinsic rewards by the standard deviation of a running discounted intrinsic
        # return. The filter steps through time, one step per row of curiosity_rewards (num_steps,
        # num_envs), carrying one running return per environment, as OpenAI's original code does.
        curiosity_reward_per_env = np.array(
            [discounted_reward.update(reward_per_step) for reward_per_step in curiosity_rewards.cpu().numpy()]
        )
        reward_rms.update(curiosity_reward_per_env.ravel())
        raw_curiosity_mean = curiosity_rewards.mean().item()
        curiosity_rewards /= np.sqrt(reward_rms.var)

        # bootstrap value if not done
        with torch.no_grad():
            next_value_ext, next_value_int = agent.get_value(next_obs)
            next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
            ext_advantages = torch.zeros_like(rewards, device=device)
            int_advantages = torch.zeros_like(curiosity_rewards, device=device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                # the intrinsic return is non-episodic: novelty beyond a lost life or game over still
                # counts, so its bootstrap is never cut (int_nextnonterminal = 1)
                int_nextnonterminal = 1.0
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                ext_advantages[t] = ext_lastgaelam = ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                int_advantages[t] = int_lastgaelam = int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)

        # one policy, one combined advantage: extrinsic weighted twice the intrinsic by default
        b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

        obs_rms.update(last_frame(b_obs).cpu().numpy().astype(np.float64))

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)

        rnd_next_obs = normalize_obs(last_frame(b_obs))

        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # RND predictor loss, on a random `update_proportion` of the minibatch so the predictor
                # sees as much data as it would with fewer environments and doesn't learn novelty away too fast
                predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs[mb_inds])
                forward_loss = F.mse_loss(predict_next_state_feature, target_next_state_feature.detach(), reduction="none").mean(-1)
                mask = (torch.rand(len(forward_loss), device=device) < args.update_proportion).float()
                forward_loss = (forward_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)

                _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss: clipped for the extrinsic head as in PPO, plain MSE for the intrinsic head
                new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                if args.clip_vloss:
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                    ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_values - b_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                    ext_v_loss = 0.5 * torch.max(ext_v_loss_unclipped, ext_v_loss_clipped).mean()
                else:
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = ext_v_loss + int_v_loss
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(combined_parameters, args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/curiosity_reward_raw", raw_curiosity_mean, global_step)
        writer.add_scalar("charts/curiosity_reward_normalized", curiosity_rewards.mean().item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/ext_value_loss", ext_v_loss.item(), global_step)
        writer.add_scalar("losses/int_value_loss", int_v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/fwd_loss", forward_loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.save_model and args.save_interval and iteration % args.save_interval == 0:
            save_checkpoint()

    if args.save_model:
        save_checkpoint()

    envs.close()
    writer.close()
