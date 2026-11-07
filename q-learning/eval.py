import random
from typing import Callable

import gymnasium as gym
import numpy as np


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    epsilon: float = 0.0,
    capture_video: bool = True,
    seed: int = 0,
):
    from algorithm import state_index

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, capture_video, run_name)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    q_table = np.load(model_path)

    obs, _ = envs.reset(seed=seed)
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_rows = q_table[state_index(obs, envs.single_observation_space)]
            actions = np.array([np.random.choice(np.flatnonzero(row == row.max())) for row in q_rows])
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos and "episode" in infos["final_info"]:
            for idx, finished in enumerate(infos["final_info"]["_episode"]):
                if finished:
                    episodic_return = infos["final_info"]["episode"]["r"][idx]
                    print(f"eval_episode={len(episodic_returns)}, episodic_return={episodic_return}")
                    episodic_returns += [episodic_return]
        obs = next_obs

    envs.close()
    return episodic_returns


if __name__ == "__main__":
    import argparse

    from algorithm import make_env

    parser = argparse.ArgumentParser(description="Greedy evaluation of a saved Q-table")
    parser.add_argument("model_path", help="a .npy Q-table saved by algorithm.py --save-model")
    parser.add_argument("--env-id", default="Taxi-v4")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--capture-video", action="store_true")
    cli = parser.parse_args()
    returns = evaluate(cli.model_path, make_env, cli.env_id, cli.eval_episodes, "eval", capture_video=cli.capture_video)
    print(f"mean return over {len(returns)} episodes: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
