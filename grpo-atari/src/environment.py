"""Atari environment factory with the standard DQN/PPO preprocessing, configured from the arguments.

Every algorithm builds its environments here, so DQN, PPO and GRPO see identical observations,
rewards and episode boundaries.
"""

import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def make_env(env_id, seed, idx, capture_video, run_name, args):
    def thunk():
        configs = dict(args.env_configs)
        if args.max_episode_steps > 0:
            # The TimeLimit wraps the raw env, inside frame skipping, so it counts emulator frames.
            configs["max_episode_steps"] = args.max_episode_steps * max(1, args.frame_skip)

        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **configs)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **configs)
        # Outside EpisodicLifeEnv and ClipRewardEnv, so logged returns are whole-game, unclipped scores.
        env = gym.wrappers.RecordEpisodeStatistics(env)

        if args.noop_max > 0:
            env = NoopResetEnv(env, noop_max=args.noop_max)
        if args.frame_skip > 1:
            env = MaxAndSkipEnv(env, skip=args.frame_skip)
        if args.episodic_life:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if args.clip_rewards:
            env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (args.screen_size, args.screen_size))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, args.frame_stack)

        env.action_space.seed(seed)
        return env

    return thunk


def make_vector_env(args, run_name, num_envs, capture_video=None, seed=None):
    capture_video = args.capture_video if capture_video is None else capture_video
    seed = args.seed if seed is None else seed
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, seed + i, i, capture_video, run_name, args) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    return envs


def finished_episodes(infos):
    """Yield (return, length) for every game that ended on this vector step (gymnasium 0.29 layout)."""
    if "final_info" not in infos:
        return
    for info in infos["final_info"]:
        if info and "episode" in info:
            yield float(info["episode"]["r"][0]), int(info["episode"]["l"][0])


def log_episodes(writer, infos, global_step):
    """Log completed games under the metric names shared by all algorithms; returns the episode returns."""
    returns = []
    for episodic_return, episodic_length in finished_episodes(infos):
        print(f"global_step={global_step}, episodic_return={episodic_return}")
        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
        writer.add_scalar("charts/episodic_length", episodic_length, global_step)
        returns.append(episodic_return)
    return returns
