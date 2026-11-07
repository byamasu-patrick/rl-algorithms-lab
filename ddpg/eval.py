from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: tuple[type[nn.Module], type[nn.Module]],
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    exploration_noise: float = 0.1,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, 0, capture_video, run_name)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    actor = Model[0](envs).to(device)
    qf = Model[1](envs).to(device)
    actor_params, qf_params = torch.load(model_path, map_location=device)
    # CleanRL's own checkpoints compute these from the vector env's action space, giving shape
    # (1, act_dim) where this project's Actor uses (act_dim,); the values are the same
    for key in ("action_scale", "action_bias"):
        if actor_params[key].dim() == 2:
            actor_params[key] = actor_params[key][0]
    actor.load_state_dict(actor_params)
    actor.eval()
    qf.load_state_dict(qf_params)
    qf.eval()
    # note: qf is not used in this script; it is loaded so the checkpoint round-trips completely

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
            actions += torch.normal(0, actor.action_scale * exploration_noise)
            actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

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
    from huggingface_hub import hf_hub_download

    from algorithm import Actor, QNetwork, make_env

    model_path = hf_hub_download(repo_id="cleanrl/Hopper-v4-ddpg_continuous_action-seed1", filename="ddpg_continuous_action.cleanrl_model")
    evaluate(
        model_path,
        make_env,
        "Hopper-v4",
        eval_episodes=10,
        run_name=f"eval",
        Model=(Actor, QNetwork),
        device="cpu",
        capture_video=False,
    )
