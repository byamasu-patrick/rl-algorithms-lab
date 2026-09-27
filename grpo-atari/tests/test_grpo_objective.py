import torch

from src.algorithms.grpo import grpo_loss, kl_estimate


def test_kl_estimate_is_zero_at_equality_and_positive_otherwise():
    logp = torch.log(torch.tensor([0.2, 0.5, 0.9]))
    assert torch.allclose(kl_estimate(logp, logp), torch.zeros(3))
    ref = torch.log(torch.tensor([0.4, 0.1, 0.3]))
    assert torch.all(kl_estimate(ref, logp) > 0)
    # Its expectation under pi_theta is the true KL(pi_theta || pi_ref).
    p, q = torch.tensor([0.7, 0.3]), torch.tensor([0.4, 0.6])
    expected = (p * (p / q).log()).sum()
    assert torch.isclose((p * kl_estimate(q.log(), p.log())).sum(), expected)


def _loss(logprob, old, adv, weights=None, ref=None, clip=0.2, beta=0.0):
    weights = torch.ones_like(logprob) if weights is None else weights
    return grpo_loss(logprob, old, adv, torch.zeros_like(logprob), weights, ref, clip, beta, 0.0)


def test_unclipped_loss_is_minus_ratio_times_advantage():
    old = torch.log(torch.tensor([0.5, 0.5]))
    new = torch.log(torch.tensor([0.55, 0.45]))
    adv = torch.tensor([1.0, -1.0])
    loss, _, _ = _loss(new, old, adv)
    assert torch.isclose(loss, torch.tensor(-((1.1 * 1.0) + (0.9 * -1.0)) / 2))


def test_ratio_is_clipped_for_positive_advantage():
    old = torch.log(torch.tensor([0.1]))
    new = torch.log(torch.tensor([0.5]))  # ratio 5
    loss, _, _ = _loss(new, old, torch.tensor([2.0]), clip=0.2)
    assert torch.isclose(loss, torch.tensor(-1.2 * 2.0))


def test_sequence_weights_average_per_trajectory_then_over_trajectories():
    # Trajectory A: 1 step with advantage 1; trajectory B: 3 steps with advantage 0. Ratio 1 everywhere.
    logp = torch.log(torch.full((4,), 0.5))
    adv = torch.tensor([1.0, 0.0, 0.0, 0.0])
    sequence = _loss(logp, logp, adv, weights=torch.tensor([1.0, 1 / 3, 1 / 3, 1 / 3]))[0]
    token = _loss(logp, logp, adv)[0]
    assert torch.isclose(sequence, torch.tensor(-0.5))   # (1/G) sum_i mean_t = (1 + 0) / 2
    assert torch.isclose(token, torch.tensor(-0.25))     # mean over 4 steps


def test_kl_penalty_is_added_with_beta():
    logp = torch.log(torch.tensor([0.5]))
    ref = torch.log(torch.tensor([0.25]))
    loss, pg_loss, kl = _loss(logp, logp, torch.tensor([0.0]), ref=ref, beta=0.1)
    assert torch.isclose(loss, pg_loss + 0.1 * kl)
    assert kl > 0


def test_gradient_increases_probability_of_advantaged_action():
    logits = torch.zeros(2, requires_grad=True)
    logp = torch.log_softmax(logits, 0)
    old = logp.detach()
    loss, _, _ = _loss(logp[:1], old[:1], torch.tensor([1.0]))
    loss.backward()
    assert logits.grad[0] < 0  # gradient descent raises logit 0, the action with positive advantage
