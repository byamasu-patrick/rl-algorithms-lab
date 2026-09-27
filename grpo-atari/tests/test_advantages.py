import numpy as np
import pytest

from src.advantages import (
    BaselineState,
    baseline_values,
    discounted_returns,
    outcome_advantages,
    process_advantages,
    sequence_weights,
)


def test_outcome_normalizes_totals_within_each_group_only():
    # Group 0 totals: 1, 3; group 1 totals: 10, 10, 40.
    rewards = [np.array([1.0, 0.0]), np.array([1.0, 1.0, 1.0]), np.array([10.0]), np.array([4.0, 6.0]), np.array([40.0])]
    groups = [0, 0, 1, 1, 1]
    adv = outcome_advantages(rewards, groups)

    g0_std = np.std([1, 3], ddof=1)
    assert np.allclose(adv[0], (1 - 2) / g0_std) and len(adv[0]) == 2
    assert np.allclose(adv[1], (3 - 2) / g0_std) and len(adv[1]) == 3
    g1 = np.array([10, 10, 40])
    assert np.allclose(adv[4], (40 - g1.mean()) / g1.std(ddof=1))
    # Every step of a trajectory shares its outcome advantage, and each group sums to zero.
    assert np.allclose(adv[3], adv[3][0])
    assert np.isclose(sum(a[0] for a in adv[2:]), 0.0, atol=1e-5)


def test_outcome_identical_scores_give_zero_advantage():
    adv = outcome_advantages([np.array([1.0]), np.array([0.0, 1.0])], [0, 0])
    assert np.allclose(np.concatenate(adv), 0.0)


def test_process_is_reward_to_go_of_group_normalized_rewards():
    rewards = [np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0])]
    adv = process_advantages(rewards, [0, 0])
    pooled = np.array([0, 1, 0, 1, 1.0])
    norm = lambda r: (r - pooled.mean()) / pooled.std(ddof=1)
    assert np.allclose(adv[0], [norm(0) + norm(1) + norm(0), norm(1) + norm(0), norm(0)])
    assert np.allclose(adv[1], [2 * norm(1), norm(1)])


def test_discounted_returns_with_bootstrap():
    assert np.allclose(discounted_returns(np.array([1.0, 0.0, 2.0]), 0.5, bootstrap=4.0),
                       [1 + 0.5 * 0 + 0.25 * 2 + 0.125 * 4, 0 + 0.5 * 2 + 0.25 * 4, 2 + 0.5 * 4])


class Args:
    baseline_constant = 2.0
    baseline_uniform_low, baseline_uniform_high = -1.0, 1.0
    baseline_ema_beta = 0.5


@pytest.mark.parametrize("kind", ["constant", "batch_mean", "same_seed_mean", "ema", "uniform", "stats"])
def test_baselines_have_one_value_per_step(kind):
    returns = [np.array([1.0, 2.0]), np.array([3.0]), np.array([5.0, 7.0])]
    out = baseline_values(kind, returns, [0, 0, 1], Args, BaselineState(), np.random.default_rng(0))
    assert [len(b) for b in out] == [2, 1, 2]
    if kind == "batch_mean":
        assert np.allclose(np.concatenate(out), np.mean([1, 2, 3, 5, 7]))
    if kind == "same_seed_mean":
        assert np.allclose(out[0], np.mean([1.5, 3.0])) and np.allclose(out[2], 6.0)
    if kind == "ema":
        # First EMA step with bias correction equals the batch mean.
        assert np.allclose(np.concatenate(out), np.mean([1, 2, 3, 5, 7]))


def test_sequence_weights_give_each_trajectory_equal_total_weight():
    weights = sequence_weights([2, 5], "sequence")
    assert np.isclose(weights[0].sum(), 1.0) and np.isclose(weights[1].sum(), 1.0)
    assert all(np.all(w == 1.0) for w in sequence_weights([2, 5], "token"))


def test_same_seed_resets_give_identical_start_states():
    # The GRPO group assumption: same seed -> same no-op start and the same first frames.
    import argparse
    from src.environment import make_env

    args = argparse.Namespace(env_configs={}, max_episode_steps=27000, frame_skip=4, noop_max=30,
                              episodic_life=False, clip_rewards=True, screen_size=84, frame_stack=4)
    env_a = make_env("BreakoutNoFrameskip-v4", 0, 1, False, "test", args)()
    env_b = make_env("BreakoutNoFrameskip-v4", 1, 2, False, "test", args)()
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    assert np.array_equal(np.asarray(obs_a), np.asarray(obs_b))
    for action in [1, 2, 3, 2, 0, 3]:
        obs_a, *_ = env_a.step(action)
        obs_b, *_ = env_b.step(action)
    assert np.array_equal(np.asarray(obs_a), np.asarray(obs_b))
    env_a.close()
    env_b.close()
