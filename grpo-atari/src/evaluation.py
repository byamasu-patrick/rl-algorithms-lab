"""Evaluation protocol shared by all algorithms, so their evaluation returns are comparable."""

from src.environment import finished_episodes, make_vector_env


def evaluate(args, run_name, act, eval_episodes, capture_video=True):
    """Play `eval_episodes` full games with `act(obs) -> actions` and return their (unclipped) scores.

    Uses the training preprocessing on a single environment, with videos in `videos/{run_name}`.
    """
    envs = make_vector_env(args, run_name, num_envs=1, capture_video=capture_video)
    obs, _ = envs.reset(seed=args.seed)
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        obs, _, _, _, infos = envs.step(act(obs))
        for episodic_return, _ in finished_episodes(infos):
            print(f"eval_episode={len(episodic_returns)}, episodic_return={episodic_return}")
            episodic_returns.append(episodic_return)
    envs.close()
    return episodic_returns[:eval_episodes]
