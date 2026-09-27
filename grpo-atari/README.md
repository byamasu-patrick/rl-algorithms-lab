# DQN vs PPO vs GRPO on Atari

Can **Group Relative Policy Optimization** learn to play Atari from pixels without a critic, and how
does it compare with the value-based and actor-critic baselines?

This project trains three algorithms on `BreakoutNoFrameskip-v4` under identical conditions:

| Algorithm | Family | Source |
| --- | --- | --- |
| **DQN** | off-policy, value-based | ported from [dqn-atari](../dqn-atari) |
| **PPO** | on-policy actor-critic | ported from [ppo-atari](../ppo-atari) |
| **GRPO** | on-policy, critic-free | the DeepSeekMath objective ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)), plus the critic-free baselines of [revisiting-grpo](../revisiting-grpo) |

All three share one environment pipeline, one run and logging setup, one evaluation protocol, and one
Hub upload path, so differences in the results come from the algorithms rather than the plumbing.
Each can run alone, or any combination can run in parallel, locally or on
[Hugging Face Jobs](#running-on-hugging-face-jobs).

---

## Layout

```text
grpo-atari/
├── algorithm.py              # train one algorithm: python algorithm.py {dqn,ppo,grpo} [flags]
├── experiments.py            # run many algorithms x seeds in parallel, locally or as HF Jobs
├── src/
│   ├── args.py               # shared flags (experiment setup, environment, budget) and the CLI
│   ├── environment.py        # Atari preprocessing and episode logging, shared by all algorithms
│   ├── run.py                # run naming, W&B/TensorBoard, seeding, checkpoints, save/evaluate/upload
│   ├── evaluation.py         # evaluation protocol shared by all algorithms
│   ├── hub.py                # Hugging Face upload and model card
│   ├── checkpoints.py        # periodic checkpoints and warm starts (from revisiting-grpo)
│   ├── replay_buffer.py      # DQN replay buffer
│   ├── advantages.py         # GRPO advantage estimators (pure functions, unit-tested)
│   └── algorithms/
│       ├── __init__.py       # registry: name -> module
│       ├── dqn.py            # QNetwork, flags, training loop
│       ├── ppo.py            # Agent (actor + critic), flags shared with GRPO, training loop
│       └── grpo.py           # Policy (actor only), group rollouts, GRPO objective, training loop
├── tests/                    # advantage estimators, GRPO objective, seeded-group determinism
├── pyproject.toml
└── requirements.txt
```

Each module in `src/algorithms` is self-contained. It declares its flags (`add_args`), checks them
(`validate_args`), and trains (`train(args, run, device)`), then calls
`run.save_evaluate_upload(...)`. Adding an algorithm means adding one module and one registry entry.
Nothing else changes.

### What the ports changed

The training loops are the same as in the source projects, with these differences:

- **Gymnasium 0.29 for PPO.** `ppo-atari` targets Gymnasium 1.x (`ALE/Breakout-v5`,
  `FrameStackObservation`, `info["_episode"]`). Here it runs on the same Gymnasium 0.29 environment
  as DQN and GRPO, so all three train on `BreakoutNoFrameskip-v4` with identical wrappers.
- **Shared plumbing.** Environment construction, episode logging, W&B/TensorBoard setup, checkpoints,
  and save/evaluate/upload moved into shared modules. PPO gained the model saving, evaluation, and
  upload that `ppo-atari` lacked.
- **PPO diagnostics.** `approx_kl`, `old_approx_kl`, and `clipfrac` are computed without gradients,
  and `clipfrac` is averaged over all minibatches instead of taken from the last one. This changes
  the logged diagnostics, not the updates.
- **Flag spellings.** `--annealing-lr` is `--anneal-lr`, and booleans use `--flag` / `--no-flag`
  (for example `--no-gae`), matching revisiting-grpo.

---

## Running

Run from this directory.

### One algorithm

```bash
python algorithm.py dqn
python algorithm.py ppo --seed 2
python algorithm.py grpo                                   # outcome supervision, 2 groups of 8
python algorithm.py grpo --advantage-type process
python algorithm.py grpo --track --save-model --upload-model --hf-entity byamasupatrick
python algorithm.py dqn --help          # the flags one algorithm accepts
tensorboard --logdir runs
```

Runs write to `runs/{exp_name}__{env_id}__{seed}/`, for example `runs/ppo__breakoutnoframeskip_v4__1/`,
with the full configuration in `config.yaml`. `--exp-name` defaults to the algorithm name. The
directory name has no timestamp, so rerunning a finished configuration exits immediately. This lets
a sweep be restarted without repeating work. Pass `-o` / `--overwrite` to rerun it anyway.

### Several algorithms and seeds

`experiments.py` expands algorithms × seeds into runs:

```bash
python experiments.py --algos dqn ppo grpo --seeds 1 2 3 --track --save-model
python experiments.py --algos dqn ppo grpo --seeds 1 2 3 --dry-run       # print the commands only
```

- **Shared flags.** Any flag `experiments.py` does not recognise is passed to every run, for example
  `--track`, `--total-timesteps`, or `--save-model`.
- **Flags for one algorithm.** These go in a quoted string, for example
  `--dqn-args "--buffer-size 500000"` or `--ppo-args "--num-envs 16"`.
- **Running locally.** Runs train as parallel processes, at most `--max-parallel` at a time (default:
  the number of CPU cores). Each run writes its output to `logs/{algo}__seed{seed}.log`, and a summary
  prints as runs finish.
- **Algorithms not ready yet.** A module can set `IMPLEMENTED = False`; `experiments.py` then skips it
  with a message instead of failing the sweep.

Size `--max-parallel` to the machine's memory. A DQN run's replay buffer holds 1M stacked frames of
4 × 84 × 84 bytes by default, about 28 GB once full. That much is needed even with
`optimize_memory_usage`, which avoids storing each next observation a second time. PPO needs well
under 1 GB. GRPO holds each iteration's episodes in memory as `uint8` frames, about 28 KB per step:
16 environments × a few thousand steps per game is a few GB. On a smaller machine, lower DQN's buffer
with `--dqn-args "--buffer-size 200000"` (about 5.6 GB).

---

## Running on Hugging Face Jobs

Add `--hf-job` to either command. With `algorithm.py` that submits one run. With `experiments.py`,
every run is submitted as its own Job, so the whole comparison trains in parallel:

```bash
python algorithm.py dqn --track --save-model --upload-model --hf-entity byamasupatrick --hf-job --hf-flavor l40sx1

python experiments.py --algos dqn ppo grpo --seeds 1 2 3 --track --save-model --upload-model \
    --hf-entity byamasupatrick --hf-job --hf-flavor l40sx1 --hf-timeout 36h
```

DQN needs a flavor with more RAM than its ~28 GB replay buffer. `l40sx1` (62 GB) fits it, while
`l4x1` (30 GB) and `t4-small` (15 GB) run out of memory before the buffer fills.

Jobs run and are billed under `baobabtech`, log to W&B, and upload models to your Hugging Face
account as `{hf_entity}/{env_id}-{exp_name}-seed{seed}`. Check a sweep first with `--hf-dry-run`.
See [hf-training-jobs](../hf-training-jobs) for the other `--hf-*` flags and how to monitor Jobs.

---

## What is shared, and so comparable

| Shared piece | Where | Effect on the comparison |
| --- | --- | --- |
| Environment | `src/environment.py` | Same wrappers, frame stack, reward clipping, and episode boundaries. |
| Budget | `--total-timesteps` (default 10M), a shared flag | Every algorithm is measured over the same number of environment steps. |
| Episode metrics | `charts/episodic_return`, `charts/episodic_length` | Whole-game, unclipped scores, logged the same way by every algorithm. |
| Evaluation | `src/evaluation.py` | `--eval-episodes` full games on one environment after training, logged as `eval/episodic_return` and `eval/mean_episodic_return`. DQN acts ε-greedily at `--end-e`; PPO and GRPO sample from their policies. |
| Tracking | `src/run.py` | W&B project `grpo-atari`, run grouped by `--exp-name` and tagged with the algorithm, so runs line up in one workspace. |

### Environment preprocessing

| Wrapper | Flag | Default |
| --- | --- | --- |
| Time limit, counted in agent steps | `--max-episode-steps` | `27000` (108k frames) |
| `RecordEpisodeStatistics` (whole-game, unclipped score) | | always |
| `NoopResetEnv` | `--noop-max` | `30` |
| `MaxAndSkipEnv` | `--frame-skip` | `4` |
| `EpisodicLifeEnv` | `--episodic-life` / `--no-episodic-life` | on for `dqn` and `ppo`; **off for `grpo`** (see [GRPO](#grpo)) |
| `FireResetEnv` | | games with a FIRE action |
| `ClipRewardEnv` | `--clip-rewards` / `--no-clip-rewards` | on |
| `ResizeObservation`, `GrayScaleObservation` | `--screen-size` | `84` |
| `FrameStack` | `--frame-stack` | `4` |

---

## GRPO

GRPO ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)) is PPO with the critic removed. There is
**no value network and no GAE**. Instead, the policy samples a *group* of outputs for the same
question, and each output's advantage is its reward relative to the rest of the group: the group mean
is subtracted, and the result is divided by the group's standard deviation.
[src/algorithms/grpo.py](src/algorithms/grpo.py) implements the objective of the paper's equation 3:

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\Big(\min\big(\rho_{i,t}\hat{A}_{i,t},\ \mathrm{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\,\hat{A}_{i,t}\big)
- \beta\, \mathbb{D}_{KL}\big[\pi_\theta \,\|\, \pi_{ref}\big]\Big)\Bigg],
\qquad \rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} \mid q, o_{i,<t})}
$$

with the KL divergence estimated per step by the paper's unbiased estimator
$\frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1 \ge 0$.

### From language models to Atari

| DeepSeekMath | Here |
| --- | --- |
| question $q$ | a start state, fixed by a reset seed |
| group of $G$ outputs | $G$ environments reset with the **same seed** (`--num-groups` groups of `--num-envs / --num-groups`) |
| output $o_i$ | one whole game |
| token $o_{i,t}$ | one action |
| reward model score $r_i$ | the game's total reward (clipped to its sign per step, as for DQN and PPO) |
| policy model | [`Policy`](src/algorithms/grpo.py): PPO's Nature CNN and actor head, with no critic head |
| reference model $\pi_{ref}$ | a frozen copy of the policy, refreshed every `--ref-update-every` iterations (Algorithm 1's outer loop) |
| $\mu$ GRPO iterations | `--update-epochs` passes over each batch |

`NoFrameskip-v4` games have no sticky actions, and the no-op start is drawn from the reset seed. So a
group starts from identical frames and diverges only through the policy's own sampling, which is what
makes the group mean a fair baseline for each member.
[tests/test_advantages.py](tests/test_advantages.py) checks this.

Each iteration plays one game in every environment. Environments are stepped individually, so one
that finishes early stops using steps while the rest of its group plays on.

### Advantages

`--advantage-type` selects how $\hat{A}_{i,t}$ is computed. None of the options uses a value function.

| `--advantage-type` | $\hat{A}_{i,t}$ | Source |
| --- | --- | --- |
| **`outcome`** (default) | $\frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})}$ for every step of game $i$, with $\mathbf{r}$ the totals of $i$'s group | DeepSeekMath §4.1.2, outcome supervision |
| `process` | rewards normalized by the mean and std of all rewards in the group, then summed from step $t$ to the end | DeepSeekMath §4.1.3, process supervision |
| `baseline` | discounted Monte Carlo return minus `--baseline-type`: `batch_mean`, `same_seed_mean`, `ema`, `stats`, `uniform`, or `constant`; divided by the batch std with `--scale-adv-batch` | revisiting-grpo's critic-free variants |

The estimators live in [src/advantages.py](src/advantages.py) as pure functions. Each one is tested
against a hand computation in [tests/test_advantages.py](tests/test_advantages.py).

If every game in a group scores the same, the group has no spread and every advantage in it is 0, so
that group teaches nothing. Early in Breakout most games score 0, so watch
`grpo/degenerate_group_fraction`. Larger groups (`--num-envs` / `--num-groups`) make identical scores
less likely.

### Loss

[`grpo_loss`](src/algorithms/grpo.py) computes the negative objective over a minibatch of steps:

- the clipped surrogate, with $\varepsilon$ = `--clip-coef`;
- plus $\beta \cdot$ KL to the reference policy, with $\beta$ = `--kl-coef` (0 disables it and the
  reference copy);
- minus `--ent-coef` × entropy.

`--loss-aggregation sequence` (the default) weights each step by $1/|o_i|$, which reproduces
$\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t$: every game counts equally, whatever its length.
`--loss-aggregation token` averages over all steps instead, so long games weigh more.
[tests/test_grpo_objective.py](tests/test_grpo_objective.py) checks the following:

- the KL estimator is zero when the policies match, positive otherwise, and in expectation equals
  the true KL;
- the clipping and both aggregations give the expected values;
- the gradient raises the probability of an action with positive advantage.

### Whole games, not lives

GRPO defaults to `--no-episodic-life`, unlike DQN and PPO. Resetting a group to a shared seed must
restart the game. With life-loss episodes, a reset in the middle of a game only continues it, so the
group members would not start from the same state. Treating whole games as episodes also follows
Machado et al. (2018), who recommend not giving the agent life-loss signals. The comparison stays
fair: every algorithm logs whole-game scores and is evaluated on whole games.

`--episodic-life` is still accepted with `--num-groups 0`. There, each episode is one life, and the
whole batch forms one unseeded group.

### Defaults compared with the paper

| | DeepSeekMath | Here | Why |
| --- | --- | --- | --- |
| Group size $G$ | 64 | 8 (`--num-envs 16`, `--num-groups 2`) | each member is a full Atari game |
| $\beta$ (`--kl-coef`) | 0.04 | 0.04 | |
| $\varepsilon$ (`--clip-coef`) | not stated | 0.1 | the Atari PPO value |
| $\mu$ (`--update-epochs`) | 1 | 4 | matches the PPO baseline; set `--update-epochs 1` for the paper's setting |
| Reference model | reset each outer iteration | refreshed every 10 iterations | training starts from a random policy, not an SFT model |
| Entropy bonus | none | 0.01 | matches the PPO baseline; `--ent-coef 0` removes it |
| Reward | learned reward model | the game score | |

### Metrics

On top of the shared episode, loss, and SPS metrics, GRPO logs:

- `losses/kl_ref`: the mean per-step KL to the reference policy;
- `grpo/advantage_mean`, `grpo/advantage_std`;
- `grpo/degenerate_group_fraction`: the fraction of groups with no score spread;
- `rollout/batch_size`, `rollout/trajectories`, `rollout/mean_trajectory_length`,
  `rollout/mean_trajectory_reward`;
- `time/collection`, `time/update`.

### Tests

```bash
pip install pytest
pytest tests
```

---

## Installation

The project needs **Python 3.10 or 3.11**. `ale-py` 0.8.1, the release that registers the Atari
environments with Gymnasium 0.29, has no wheels for newer Pythons.

```powershell
py install 3.11                # once
py -V:3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the [Hugging Face Jobs launcher](../hf-training-jobs),
`hf-jobs-launch`, from GitHub, pinned to the `hf-jobs-launch-v0.1.0` tag. To work on the launcher
itself, run `pip install -e ../hf-training-jobs` afterwards.

| Pin | Reason |
| --- | --- |
| `gymnasium[atari,accept-rom-license]==0.29.1` | Same Gymnasium as revisiting-grpo and dqn-atari; the extras install `shimmy`, `ale-py`, and the ROMs. |
| `ale-py==0.8.1` | Registers the `*NoFrameskip-v4` environments with Gymnasium 0.29. Limits Python to ≤ 3.11. |
| `numpy==1.26.4` | Gymnasium 0.29's environment checker uses `np.bool8`, which NumPy 2 removed. |
| `stable-baselines3==2.3.2` | Atari wrappers; last line compatible with Gymnasium 0.29. |
| `opencv-python-headless` | Needed by `ResizeObservation`, without the `libGL` dependency missing from the Jobs image. |
| `moviepy==1.0.3` | Gymnasium 0.29's `RecordVideo` is written against the MoviePy 1.x API. |
| `PyYAML`, `tqdm` | `config.yaml` in each run directory, and progress reporting. |
| `huggingface_hub`, `tenacity` | Model upload with retries. |
| `torch==2.13.0` | The Windows pip wheel is CPU-only; the Linux wheels used in Jobs include CUDA. |

---

## Command-line reference

`python algorithm.py <algo> --help` lists every flag an algorithm accepts. Boolean flags take a
`--no-` form, for example `--no-track`.

### Shared by all algorithms

| Flag | Default | Description |
| --- | --- | --- |
| `--exp-name` | algorithm name | Experiment name; W&B group, run directory, and model repository name. |
| `--run-name` | `{exp_name}__{env_id}__{seed}` | Run directory and W&B run name. |
| `--seed` | `1` | Seeds Python, NumPy, Torch, and the environments. |
| `--torch-deterministic` | on | Sets `torch.backends.cudnn.deterministic`. |
| `--cuda` | on | Use CUDA when available. |
| `--track` | off | Mirror metrics to Weights & Biases. |
| `--wandb-project-name` | `grpo-atari` | W&B project. |
| `--wandb-entity` | `mscsr000324-must` | W&B team/entity. |
| `--capture-video` | off | Record training episodes from the first environment to `videos/{run_name}`. |
| `--num-checkpoints` | `10` | Checkpoints saved over the run to `runs/{run_name}/checkpoints/`; `0` disables. |
| `--checkpoint-every` | `0` | Env steps between checkpoints, when `--num-checkpoints` is `0`. |
| `--checkpoint-load-path` | `None` | Initialize weights and optimizer from a checkpoint (a warm start: step counters restart at 0). |
| `--checkpoint-param-filters` | `None` | JSON `{object: regex}`; only matching parameters are loaded. |
| `--log-every` | `0` | Env steps between detailed stat logs; `0` logs every iteration. |
| `-o`, `--overwrite` | off | Replace an existing run directory instead of exiting. |
| `--save-model` | off | Save the final model, then evaluate it. |
| `--eval-episodes` | `10` | Evaluation games played on the saved model. |
| `--upload-model` | off | Push the model to the Hub. Requires `--save-model`. |
| `--hf-entity` | `""` | Hub user or org for the model repository; empty means the token's account. |
| `--total-timesteps` | `10000000` | Environment steps, the same budget for every algorithm. |
| `--env-id`, `--env-configs`, `--noop-max`, `--frame-skip`, `--frame-stack`, `--screen-size`, `--episodic-life`, `--clip-rewards`, `--max-episode-steps` | see [preprocessing](#environment-preprocessing) | Environment. |

### `dqn`

| Flag | Default | Description |
| --- | --- | --- |
| `--learning-rate` | `1e-4` | Adam learning rate. |
| `--num-envs` | `1` | Parallel environments. |
| `--buffer-size` | `1000000` | Replay memory capacity in transitions. |
| `--gamma` | `0.99` | Discount factor. |
| `--tau` | `1.0` | Target network update rate; `1.0` is a hard copy. |
| `--target-network-frequency` | `1000` | Steps between target network updates. |
| `--batch-size` | `32` | Transitions sampled per update. |
| `--start-e`, `--end-e` | `1`, `0.01` | Epsilon schedule endpoints. |
| `--exploration-fraction` | `0.10` | Fraction of `--total-timesteps` over which epsilon is annealed. |
| `--learning-starts` | `80000` | Step at which gradient updates begin. |
| `--train-frequency` | `4` | Steps between gradient updates. |

### `ppo`

| Flag | Default | Description |
| --- | --- | --- |
| `--learning-rate` | `2.5e-4` | Adam learning rate. |
| `--num-envs` | `8` | Parallel environments. |
| `--num-steps` | `128` | Steps per environment per rollout. |
| `--anneal-lr` | on | Linearly anneal the learning rate to 0. |
| `--gamma` | `0.99` | Discount factor. |
| `--gae` | on | GAE advantages; `--no-gae` uses bootstrapped discounted returns minus values. |
| `--gae-lambda` | `0.95` | GAE lambda. |
| `--num-minibatches` | `4` | Minibatches per epoch. |
| `--update-epochs` | `4` | Epochs over each batch. |
| `--norm-adv` | on | Normalize advantages within each minibatch. |
| `--clip-coef` | `0.1` | Surrogate and value clipping coefficient. |
| `--clip-vloss` | on | Clip the value loss. |
| `--ent-coef` | `0.01` | Entropy bonus coefficient. |
| `--vf-coef` | `0.5` | Value loss coefficient. |
| `--max-grad-norm` | `0.5` | Gradient clipping norm. |
| `--target-kl` | `None` | Stop the epoch loop early above this approximate KL. |

### `grpo`

Shares the policy-optimization flags of `ppo` (`--learning-rate`, `--anneal-lr`, `--gamma`,
`--num-minibatches`, `--update-epochs`, `--norm-adv`, `--clip-coef`, `--ent-coef`, `--max-grad-norm`,
`--target-kl`). It has none of PPO's critic flags (`--gae`, `--gae-lambda`, `--vf-coef`,
`--clip-vloss`), and its own defaults and flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--num-envs` | `16` | Environments; each plays one game per iteration. |
| `--num-steps` | `0` | `0` collects whole games. A positive value collects fixed-length segments, with no bootstrap (there is no critic); only with `--advantage-type baseline`. |
| `--norm-adv` | off | Advantages are already group-normalized. |
| `--advantage-type` | `outcome` | `outcome`, `process`, or `baseline`; see [Advantages](#advantages). |
| `--num-groups` | `2` | Groups of environments sharing a reset seed; each needs at least 2 members. `0`: the whole batch is one unseeded group. |
| `--kl-coef` | `0.04` | β, the weight of the KL penalty to the reference policy; `0` disables it. |
| `--ref-update-every` | `10` | Iterations between copies of the policy into the reference; `0` keeps the initial policy. |
| `--loss-aggregation` | `sequence` | `sequence`: 1/G Σᵢ 1/\|oᵢ\| Σₜ, as in the paper; `token`: mean over all steps. |
| `--baseline-type` | `batch_mean` | With `--advantage-type baseline`: `batch_mean`, `same_seed_mean`, `ema`, `stats`, `uniform`, or `constant`. |
| `--scale-adv-batch` | on | With `--advantage-type baseline`: divide advantages by the batch std. |
| `--baseline-constant` | `None` | Baseline for `constant`. |
| `--baseline-uniform-low`, `--baseline-uniform-high` | `None` | Range for `uniform`; falls back to the env's reward range, then `[-1, 1]`. |
| `--baseline-ema-beta` | `0.9` | Decay for `ema`, with Adam-style bias correction. |
| `--episodic-life` | **off** | See [Whole games, not lives](#whole-games-not-lives). |

---

## References

- Mnih et al., *Human-level control through deep reinforcement learning* (2015).
  [Nature 518, 529–533](https://www.nature.com/articles/nature14236)
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017).
  [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*
  (2024), which introduced GRPO. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- de Oliveira et al., *Learning Without Critics? Revisiting GRPO in Classical Reinforcement Learning
  Environments*. Latinx in AI @ NeurIPS 2025. Code and citation in [revisiting-grpo](../revisiting-grpo).
- Schulman, *Approximating KL Divergence* (2020), the KL estimator used in the objective.
  [joschu.net/blog/kl-approx.html](http://joschu.net/blog/kl-approx.html)
- Machado et al., *Revisiting the Arcade Learning Environment* (2018), on not using life-loss signals.
  [arXiv:1709.06009](https://arxiv.org/abs/1709.06009)

## License

MIT, see [LICENSE](../LICENSE) at the repository root. Provenance for the repository as a whole is
recorded in [NOTICE](../NOTICE).
