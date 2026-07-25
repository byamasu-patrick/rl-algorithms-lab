"""
Rollout collection functions for environment interaction.

"""

import torch
import numpy as np
from src.utils.logger import log_stats


def collect_fixed_rollout(agent, writer, dtype, device, envs, obs, dones, values, actions, logprobs,
                        next_obs, next_done, rewards, return_deque, length_deque, global_step, args):
    """
    Collect trajectories for a fixed number of environment steps.
    Args:
        agent: The reinforcement learning agent.
        writer: TensorBoard summary writer.
        dtype: Data type for the tensors.
        device: Device for the tensors.
        envs: Vectorized environments.
        obs: Tensor to store observations.
        dones: Tensor to store done flags.
        values: Tensor to store value estimates.
        actions: Tensor to store actions taken.
        logprobs: Tensor to store log probabilities of actions.
        next_obs: Tensor to store the next observations.
        next_done: Tensor to store the next done flags.
        rewards: Tensor to store rewards received.
        return_deque: Deque to store episodic returns.
        length_deque: Deque to store episodic lengths.
        global_step: Current global step count.
        args: Command-line arguments containing hyperparameters.
    """    
    for step in range(args.num_steps):
        global_step += args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        next_done = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward, dtype=dtype).to(device).view(-1)
        next_obs, next_done = torch.tensor(next_obs_np, dtype=dtype).to(device), torch.tensor(next_done, dtype=dtype).to(device)

        log_stats(infos, writer, global_step, return_deque, length_deque)

    return (obs, actions, logprobs, rewards, values, next_obs, next_done, global_step)

def collect_episode_rollout(agent, writer, dtype, device, envs, return_deque, length_deque, next_obs, global_step, args):
    """
    Collect complete episodes from vectorized environments.
    Args:
        agent: The reinforcement learning agent.
        writer: TensorBoard summary writer.
        dtype: Data type for the tensors.
        device: Device for the tensors.
        envs: Vectorized environments.
        return_deque: Deque to store episodic returns.
        length_deque: Deque to store episodic lengths.
        next_obs: Tensor to store the next observations.
        global_step: Current global step count.
        args: Command-line arguments containing hyperparameters.
    """
    # We collect per env first so indexes match between obs, actions, and returns across episodes
    per_env_obs = [[] for _ in range(args.num_envs)]
    per_env_actions = [[] for _ in range(args.num_envs)]
    per_env_logprobs = [[] for _ in range(args.num_envs)]
    per_env_rewards = [[] for _ in range(args.num_envs)]
    per_env_values = [[] for _ in range(args.num_envs)]
    finished = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

    while not bool(finished.all()):
        # Count only steps from envs still collecting
        global_step += int((~finished).sum().item())
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
        next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        done = np.logical_or(terminations, truncations)
        reward_t = torch.tensor(reward, dtype=dtype, device=device).view(-1)

        # Record for envs still collecting
        for i in range(args.num_envs):
            if not bool(finished[i]):
                per_env_obs[i].append(next_obs[i].detach())
                per_env_actions[i].append(action[i].detach())
                per_env_logprobs[i].append(logprob[i].detach())
                per_env_rewards[i].append(reward_t[i].detach())
                per_env_values[i].append(value.flatten()[i].detach())
                if bool(done[i]):
                    finished[i] = True

        log_stats(infos, writer, global_step, return_deque, length_deque)
        next_obs = torch.tensor(next_obs_np, dtype=dtype, device=device)

    # Build tensors for episode mode
    b_obs_list = []
    b_actions_list = []
    b_logprobs_list = []
    b_values_list = []
    per_env_rewards_list = []  # Keep raw rewards for return calculation
    per_env_values_list = []   # Keep values for TD-n in episode mode

    for i in range(args.num_envs):
        b_obs_list.extend(per_env_obs[i])
        b_actions_list.extend(per_env_actions[i])
        b_logprobs_list.extend(per_env_logprobs[i])
        b_values_list.extend(per_env_values[i])
        per_env_rewards_list.append([float(r) for r in per_env_rewards[i]])
        per_env_values_list.append([float(v) for v in per_env_values[i]])

    b_obs = torch.stack(b_obs_list, dim=0)
    b_actions = torch.stack(b_actions_list, dim=0)
    b_logprobs = torch.stack(b_logprobs_list, dim=0)
    b_values = torch.stack(b_values_list, dim=0)

    return (b_obs, b_actions, b_logprobs, b_values, per_env_rewards_list, global_step, next_obs, per_env_values_list)

def flatten_rollout_batch(obs, logprobs, actions, values, rewards, envs, args):
    """
    Convert collected trajectory lists into batched tensors
    used by return calculation and optimization.
    Args:
        obs: List of observations collected during rollout.
        logprobs: List of log probabilities of actions collected during rollout.
        actions: List of actions taken during rollout.
        values: List of value estimates collected during rollout.
        rewards: List of rewards received during rollout.
        envs: Vectorized environments.
        args: Command-line arguments containing hyperparameters.
    """
    # Flatten the batch for fixed rollouts
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_values = values.reshape(-1)
    b_rewards = rewards.reshape(-1)
    per_env_rewards_list = [b_rewards[i*args.num_steps:(i+1)*args.num_steps] for i in range(args.num_envs)]

    return (b_obs, b_logprobs, b_actions, b_values, b_rewards, per_env_rewards_list)
