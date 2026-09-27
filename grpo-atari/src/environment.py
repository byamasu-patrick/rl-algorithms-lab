"""Atari environment factory with the standard DQN/PPO preprocessing, configured from the arguments."""

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
