
"""Training utilities for reinforcement learning experiments."""
import torch


def compute_policy_loss(mb_advantages, ratio, args):
    """
    Compute the policy loss for the reinforcement learning agent.
    Args:
        mb_advantages (torch.Tensor): Advantages for the minibatch.
        ratio (torch.Tensor): Ratio of new and old policy probabilities.
        args: Command-line arguments containing hyperparameters.
    
    """    
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    return pg_loss

def compute_value_loss(mb_advantages, ratio, newvalue, b_returns, mb_inds, b_values, entropy, args, dtype, device):
    """
    Compute the value loss for the reinforcement learning agent.
    Args:
        mb_advantages (torch.Tensor): Advantages for the minibatch.
        ratio (torch.Tensor): Ratio of new and old policy probabilities.
        newvalue (torch.Tensor): New value estimates from the critic.
        b_returns (torch.Tensor): Returns
        mb_inds (torch.Tensor): Indices for the minibatch.
        b_values (torch.Tensor): Old value estimates from the critic.
        entropy (torch.Tensor): Entropy of the policy.
        args: Command-line arguments containing hyperparameters.
        dtype: Data type for the tensors.
        device: Device for the tensors.
    """

    pg_loss = compute_policy_loss(mb_advantages, ratio, args)
    
    newvalue = newvalue.view(-1)
    if args.use_value_fn:
        if args.clip_vloss:
            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
            v_clipped = b_values[mb_inds] + torch.clamp(
                newvalue - b_values[mb_inds],
                -args.clip_coef,
                args.clip_coef,
            )
            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
    else:
        v_loss = torch.tensor(0.0, dtype=dtype, device=device)

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    return (loss, v_loss, pg_loss, entropy_loss)

def compute_explained_variance():

    pass

def training_minibatch(policy_ratios, b_advantages, start, minibatch_size, b_inds, clipfracs, agent, b_obs, b_actions, b_logprobs, b_returns, b_values, dtype, device, args):
    """
    Perform a training step for a minibatch of data.
    Args:
        policy_ratios (list): List to store policy ratios for analysis.
        b_advantages (torch.Tensor): Advantages for the minibatch.
        start (int): Starting index for the minibatch.
        minibatch_size (int): Size of the minibatch.
        b_inds (torch.Tensor): Indices for the minibatch.
        clipfracs (list): List to store clipping fractions for analysis.
        agent (Agent): The reinforcement learning agent.
        b_obs (torch.Tensor): Observations for the minibatch.
        b_actions (torch.Tensor): Actions taken in the minibatch.
        b_logprobs (torch.Tensor): Log probabilities of actions in the minibatch.
        b_returns (torch.Tensor): Returns for the minibatch.
        b_values (torch.Tensor): Value estimates for the minibatch.
        dtype: Data type for the tensors.
        device: Device for the tensors.
        args: Command-line arguments containing hyperparameters.
    """
    end = start + minibatch_size
    mb_inds = b_inds[start:end]

    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
    logratio = newlogprob - b_logprobs[mb_inds]
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
        policy_ratios.append(ratio.mean().item())

    mb_advantages = b_advantages[mb_inds]
    if args.norm_adv:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)


    (loss, v_loss, pg_loss, entropy_loss) = compute_value_loss(mb_advantages, ratio, newvalue, b_returns, mb_inds, b_values, entropy, args, dtype, device)

    return (loss, v_loss, pg_loss, entropy_loss, approx_kl, old_approx_kl, clipfracs)
