# PPO with Continuous Actions

PPO for continuous control, on the **PyBullet** robotics environments. The default task is
`HalfCheetahBulletEnv-v0`.

The learning algorithm is unchanged from [../ppo/](../ppo/). Two things change: the policy becomes a
diagonal Gaussian over a real-valued action vector instead of a categorical distribution over
discrete actions, and the environment gets running observation and reward normalization, which
continuous control needs and classic control does not.

---

## Relationship to `ppo/`

The GAE recursion, the clipped surrogate, the clipped value loss, and the minibatch loop are all
identical. For the algorithm itself, read **[../ppo/README.md](../ppo/README.md)**. This document
covers only the differences.

| | `ppo/` | here |
| --- | --- | --- |
| Action space | `Discrete` | `Box`, asserted at [algorithm.py:139](algorithm.py#L139) |
| Policy head | logits into `Categorical` | mean vector plus learned log-std into `Normal` |
| Log-prob | scalar per step | per-dimension, summed over the action vector |
| Default env | `CartPole-v1` | `HalfCheetahBulletEnv-v0` |
| Wrappers | `RecordEpisodeStatistics` only | plus action clipping and observation/reward normalization |
| Entropy bonus | 0.01 | 0.0 |

The critic is untouched: the same `obs -> 64 -> 64 -> 1` Tanh MLP, still a separate trunk from the
actor.

---

## Layout

```text
ppo-continuous-actions/
├── algorithm.py       # argument parsing, env construction, training loop
├── src/
│   └── agent.py       # Agent: Gaussian policy and value MLPs
├── pyproject.toml     # Poetry dependency declaration (complete)
└── requirements.txt   # pip freeze, missing torch (see Installation)
```

---

## Installation

Use **Poetry**. It is the only complete dependency list for this project:

```bash
cd ppo-continuous-actions
poetry install
```

> **`requirements.txt` is incomplete here.** It was frozen from an environment that did not have
> PyTorch, so it lists only `gymnasium`, `numpy`, `pybullet`, and `pybullet_envs_gymnasium` (plus
> their transitive deps). `pip install -r requirements.txt` produces an environment where
> `import torch` fails. `pyproject.toml` adds `torch`, `tensorboard`, `wandb`, `moviepy`, and
> `pygame` at the versions the sibling PPO projects pin, and is marked accordingly.

Core pins are `torch` 2.13.0 and `gymnasium` 1.3.0, matching `ppo/`.

The import is what registers the `*BulletEnv-v0` ids with Gymnasium, so it looks unused to a linter
but is load-bearing. PyBullet is used rather than MuJoCo because it is open source and needs no
licence; the trade-off is that **scores are not comparable to MuJoCo `HalfCheetah-v4`**, since the
dynamics and reward scale differ.

---

## Quick start

```bash
cd ppo-continuous-actions
python algorithm.py                          # HalfCheetahBulletEnv-v0, 2M steps
tensorboard --logdir runs
```

Run it from this directory, since it imports `from src.agent import Agent`.

```bash
python algorithm.py --gym-id Walker2DBulletEnv-v0 --seed 3
python algorithm.py --total-timesteps 50000   # short plumbing check
```

---

## The Gaussian policy

[src/agent.py](src/agent.py) replaces the categorical head with a diagonal Gaussian:

```text
actor_mean:   obs -> 64 -> 64 -> act_dim,  Tanh,  final layer orthogonal gain 0.01
actor_logstd: nn.Parameter(zeros(1, act_dim))          <- state-independent, learned
critic:       obs -> 64 -> 64 -> 1,        Tanh,  final layer orthogonal gain 1.0
```

Four points worth understanding:

- **The log-std is a free parameter, not a network output.** It does not depend on the observation.
  One spread per action dimension is learned for the whole state space, which is the standard choice
  for on-policy continuous control: it is far easier to optimize than a state-conditioned variance,
  which tends to collapse early. It is stored as a log so the exponential keeps the standard
  deviation positive without a constraint.
- **It starts at zero, so the initial standard deviation is `exp(0) = 1`.** Combined with the `0.01`
  gain on the mean head, the policy begins as near-zero-mean unit-variance noise, which is broad
  exploration without a directional bias.
- **Log-probs and entropy are summed over the action dimensions**
  ([src/agent.py:46](src/agent.py#L46)). A diagonal Gaussian treats the dimensions as independent,
  so the joint log-probability is the sum of the per-dimension ones. Omitting the sum would leave a
  per-dimension tensor where the ratio needs one number per timestep.
- **The mean is unbounded and there is no squashing.** Samples can fall outside the action range,
  and the `ClipAction` wrapper is what bounds them. The importance ratio is computed from the
  log-probability of the *unclipped* sample, so the density is slightly mismatched to what the
  environment actually executed. This is what the reference implementations do, and it is a known
  approximation rather than an oversight.

As the policy improves, the learned log-std drifts downward on its own, which is why no entropy
bonus is used here.

---

## Observation and reward normalization

Continuous-control observations arrive on wildly different scales (joint angles near 1, velocities
in the tens), and PyBullet rewards are unbounded. Five wrappers handle that
([algorithm.py:123-127](algorithm.py#L123-L127)), applied in this order:

| Order | Wrapper | Effect |
| --- | --- | --- |
| 1 | `RecordEpisodeStatistics` | innermost, so it logs **raw** returns |
| 2 | `ClipAction` | clips sampled actions into the valid `Box` range |
| 3 | `NormalizeObservation` | running mean/variance standardization |
| 4 | `TransformObservation` | clips normalized observations to `[-10, 10]` |
| 5 | `NormalizeReward` | scales rewards by a discounted running return std |
| 6 | `TransformReward` | clips scaled rewards to `[-10, 10]` |

`RecordEpisodeStatistics` being applied first matters: it sits inside the normalization wrappers, so
`charts/episodic_return` is the true environment return and stays comparable across runs, while the
agent trains on the normalized signal. The two are on different scales and should not be compared to
each other.

The clipping steps exist because a running normalizer is unreliable early on, when few samples have
been seen, and an outlier can otherwise produce an observation or reward large enough to destabilize
the update.

---

## Hyperparameters

Eight defaults differ from `ppo/`. Everything else is identical, and the full flag reference is in
[../ppo/README.md](../ppo/README.md#command-line-reference).

| Flag | `ppo/` | here | Why |
| --- | --- | --- | --- |
| `--gym-id` | `CartPole-v1` | `HalfCheetahBulletEnv-v0` | continuous control task |
| `--learning-rate` | 2.5e-4 | 3e-4 | the standard continuous-control rate |
| `--total-timesteps` | 25,000 | 2,000,000 | locomotion needs millions of steps |
| `--num-envs` | 4 | 1 | a single env, with a long rollout instead |
| `--num-steps` | 128 | 2048 | long rollouts give GAE a better horizon |
| `--num-minibatches` | 4 | 32 | keeps minibatches small despite the large batch |
| `--update-epochs` | 4 | 10 | more reuse per batch, since samples are expensive |
| `--ent-coef` | 0.01 | 0.0 | the learned log-std already controls exploration |

Together these are the conventional PPO continuous-control preset rather than independent choices.
Derived values:

```text
batch_size       = num_envs × num_steps          # 1 × 2048 = 2048
minibatch_size   = batch_size // num_minibatches # 2048 // 32 = 64
num_updates      = total_timesteps // batch_size # 2000000 // 2048 = 976
gradient_steps   = num_updates × update_epochs × num_minibatches   # 976 × 10 × 32 = 312320
actual env steps = num_updates × batch_size      # 1998848
```

Note that `--num-envs 1` means rollout collection is fully serial, so throughput is bounded by the
simulator. Raising it shortens wall-clock time but changes the batch size, and therefore the
effective hyperparameters.

---

## Known issues

**`TransformObservation` is called with too few arguments.**
[algorithm.py:125](algorithm.py#L125) reads:

```python
env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
```

In Gymnasium 1.x the signature is `TransformObservation(env, func, observation_space)`, and
`observation_space` has **no default value**. As written this raises `TypeError: __init__() missing
1 required positional argument: 'observation_space'` while building the environment, before training
starts. The two-argument form was valid in the original `gym`, which is where this pattern comes
from. The fix:

```python
env = gym.wrappers.TransformObservation(
    env, lambda obs: np.clip(obs, -10, 10), env.observation_space
)
```

**Dependencies are missing from both dependency files.** See [Installation](#installation).
`pybullet_envs_gymnasium` is imported but not declared anywhere.

**Inherited from `ppo/`,** since the training loop is unchanged. Described in
[../ppo/README.md](../ppo/README.md#scope-and-known-deviations): `approx_kl` and `clipfrac`
computed inside the autograd graph, `--target-kl` only breaking after a full epoch, `terminated`
and `truncated` merged into one `done` flag, and `clipfracs` reassigned rather than appended. The
merged-done issue matters more here, because PyBullet locomotion episodes hit their time limit
routinely, and each truncation is treated as a true terminal state.

---

## References

- Schulman et al., *Proximal Policy Optimization Algorithms* (2017). [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015). [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
- Andrychowicz et al., *What Matters In On-Policy Reinforcement Learning?* (2020). [arXiv:2006.05990](https://arxiv.org/abs/2006.05990)
- Huang et al., *The 37 Implementation Details of Proximal Policy Optimization* (2022). [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

## License

MIT, see [LICENSE](../LICENSE) at the repository root. This directory is original work;
provenance for the repository as a whole is recorded in [NOTICE](../NOTICE).
