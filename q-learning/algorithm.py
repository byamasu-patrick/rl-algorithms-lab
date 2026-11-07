import argparse
import os
import time
import random
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument("--exp-name", type=str, default="q_learning",
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to save the Q-table into the `runs/{run_name}` folder")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Taxi-v4",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=200000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="the step size alpha of the Q-learning update")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--start-e", type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
                        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")

    return parser.parse_args()


def make_env(env_id, seed, idx, capture_video, run_name):

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env, stats_key="episode")
        env.action_space.seed(seed)
        return env
    return thunk


# The Q-table is indexed by an integer state. Discrete observations already are one; a Tuple of
# Discrete spaces (Blackjack) is flattened row-major into a single index.
def state_space_size(observation_space):
    if isinstance(observation_space, gym.spaces.Discrete):
        return int(observation_space.n)
    if isinstance(observation_space, gym.spaces.Tuple) and all(isinstance(s, gym.spaces.Discrete) for s in observation_space):
        return int(np.prod([s.n for s in observation_space]))
    raise ValueError(f"tabular Q-learning needs a Discrete (or Tuple of Discrete) observation space, got {observation_space}")


def state_index(obs, observation_space):
    if isinstance(observation_space, gym.spaces.Discrete):
        return np.asarray(obs, dtype=np.int64) - observation_space.start
    # a vector env batches a Tuple observation as a tuple of arrays, one per component
    dims = [s.n for s in observation_space]
    components = [np.asarray(o, dtype=np.int64) - s.start for o, s in zip(obs, observation_space)]
    return np.ravel_multi_index(components, dims)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = parse_args()

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

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

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
         for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    n_states = state_space_size(envs.single_observation_space)
    n_actions = int(envs.single_action_space.n)
    print("states:", n_states, "actions:", n_actions)

    # Agent initialization: one value per (state, action) pair, starting at zero
    q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        states = state_index(obs, envs.single_observation_space)

        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # argmax with random tie-breaking, so an all-zero row doesn't always pick action 0
            q_rows = q_table[states]
            actions = np.array([np.random.choice(np.flatnonzero(row == row.max())) for row in q_rows])

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Episode statistics. SyncVectorEnv flattens the nested `episode` dict into
        # per-key arrays, each with a matching `_key` boolean mask over environments.
        if "final_info" in infos and "episode" in infos["final_info"]:
            episode = infos["final_info"]["episode"]
            for idx, finished in enumerate(infos["final_info"]["_episode"]):
                if finished:
                    episodic_return = episode["r"][idx]
                    episodic_length = episode["l"][idx]
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)

        # Under SAME_STEP autoreset `next_obs` already holds the observation from the fresh
        # episode, so the true final state has to be read back out of `final_obs`.
        real_next_obs = next_obs
        if "final_obs" in infos:
            real_next_obs = np.array(next_obs, copy=True)
            for idx, final_obs in enumerate(infos["final_obs"]):
                if final_obs is not None:
                    if isinstance(envs.single_observation_space, gym.spaces.Tuple):
                        for component, value in zip(real_next_obs, final_obs):
                            component[idx] = value
                    else:
                        real_next_obs[idx] = final_obs
        next_states = state_index(real_next_obs, envs.single_observation_space)

        # ALGO LOGIC: training. One tabular TD update per transition, no replay and no target table.
        td_target = rewards + args.gamma * q_table[next_states].max(axis=1) * (1 - terminations)
        td_error = td_target - q_table[states, actions]
        q_table[states, actions] += args.learning_rate * td_error

        if global_step % 100 == 0:
            writer.add_scalar("losses/td_error", np.abs(td_error).mean(), global_step)
            writer.add_scalar("losses/q_values", q_table[states, actions].mean(), global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

    print("SPS:", int(args.total_timesteps / (time.time() - start_time)))

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.npy"
        np.save(model_path, q_table)
        print(f"model saved to {model_path}")
        from eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            epsilon=0.0,
            capture_video=args.capture_video,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
