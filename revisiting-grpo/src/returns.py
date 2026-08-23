import torch

def calculate_gae_returns(
    agent,
    next_obs,
    next_done,
    rewards,
    values,
    dones,
    args,
    per_env_returns,
    b_returns_list,
    device
):
    """
    Calculate Generalized Advantage Estimation (GAE).

    Computes:
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        A_t = delta_t + gamma * lambda * A_{t+1}

        Return_t = A_t + V(s_t)

    Used for PPO/A2C-style algorithms.

    Args:
        agent: The reinforcement learning agent.
        next_obs: The next observations after the rollout.
        next_done: The done flags for the next observations.
        rewards: The rewards collected during the rollout.
        values: The value estimates collected during the rollout.
        dones: The done flags collected during the rollout.
        args: Command-line arguments containing hyperparameters.
        per_env_returns: List to store returns grouped by environment.
        b_returns_list: List to store flattened return targets.
        device: Device for the tensors.

    Returns:
        b_returns:
            Flattened return targets.
        b_advantages:
            Flattened advantage estimates.
        per_env_returns:
            Returns grouped by environment.
    """
    
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
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
        b_returns = (advantages + values).reshape(-1)
        b_advantages = advantages.reshape(-1)

    # Populate helpers for consistency
    for env_idx in range(args.num_envs):
        env_rets = (advantages[:, env_idx] + values[:, env_idx]).detach().cpu().numpy().tolist()
        per_env_returns.append(env_rets)
        b_returns_list.extend(env_rets)


    return b_returns, b_advantages, per_env_returns


def calculate_td_returns(
    agent,
    next_obs,
    next_done,
    rewards,
    dones,
    args,
    values,
    per_env_returns,
    b_returns_list,
    dtype,
    device
):
    """
    Calculate n-step Temporal Difference returns.

    Computes:

        G_t = r_t + gamma*r_{t+1} + ... + gamma^n*V(s_{t+n})

    If an episode terminates before n steps,
    bootstrapping is stopped.

    Args:
        agent: The reinforcement learning agent.
        next_obs: The next observations after the rollout.
        next_done: The done flags for the next observations.
        rewards: The rewards collected during the rollout.
        dones: The done flags collected during the rollout.
        args: Command-line arguments containing hyperparameters.
        values: The value estimates collected during the rollout.
        per_env_returns: List to store returns grouped by environment.
        b_returns_list: List to store flattened return targets.
        dtype: Data type for the tensors.
        device: Device for the tensors.

    Returns:
        b_returns:
            Flattened TD return targets.
        per_env_returns:
            Returns grouped by environment.
    """
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
    rewards_np = rewards.detach().cpu().numpy()
    values_np = values.detach().cpu().numpy()
    dones_np = dones.detach().cpu().numpy()
    next_done_np = next_done.detach().cpu().numpy()
    next_value_np = next_value.detach().cpu().numpy()[0]
    
    for env_idx in range(args.num_envs):
        env_returns = []
        for t in range(args.num_steps):
            G = 0.0
            gamma_pow = 1.0
            # Sum rewards from t to num_steps-1
            for k in range(args.num_steps - t):
                t_k = t + k
                r_tk = float(rewards_np[t_k][env_idx])
                G += gamma_pow * r_tk
                gamma_pow *= args.gamma
                if dones_np[t_k][env_idx]:
                    break
            else:
                # If we didn't break (no done), add bootstrap value
                next_v = float(next_value_np[env_idx])
                nonterm = float(1.0 - next_done_np[env_idx])
                G += gamma_pow * nonterm * next_v
            env_returns.append(G)
        per_env_returns.append(env_returns)
        b_returns_list.extend(env_returns)
    b_returns = torch.tensor(b_returns_list, dtype=dtype, device=device)

    return b_returns, per_env_returns


def calculate_mc_returns(
    per_env_rewards_list,
    args,
    per_env_returns,
    b_returns_list,
    dtype,
    device
):
    """
    Calculate Monte Carlo returns.

    Computes:

        G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2}+ ...

    No value-function bootstrapping is used.

    Args:
        per_env_rewards_list: List of rewards grouped by environment.
        args: Command-line arguments containing hyperparameters.
        per_env_returns: List to store returns grouped by environment.
        b_returns_list: List to store flattened return targets.
        dtype: Data type for the tensors.
        device: Device for the tensors.

    Returns:
        b_returns:
            Flattened return targets.
        per_env_returns:
            Returns grouped by environment.
    """
    for env_rewards in per_env_rewards_list:
        R = 0.0
        env_returns = []
        for r in reversed(env_rewards):
            R = float(r) + args.gamma * R
            env_returns.append(R)
        env_returns.reverse()
        per_env_returns.append(env_returns)
        b_returns_list.extend(env_returns)
    b_returns = torch.tensor(b_returns_list, dtype=dtype, device=device)

    return (b_returns, per_env_returns, b_returns_list)
