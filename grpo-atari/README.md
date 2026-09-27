# DQN vs PPO vs GRPO on Atari

Can **Group Relative Policy Optimization** learn to play Atari from pixels without a critic, and how
does it compare with the value-based and actor-critic baselines?

This project trains three algorithms on `BreakoutNoFrameskip-v4` under identical conditions:

| Algorithm | Family | Source |
| --- | --- | --- |
| **DQN** | off-policy, value-based | ported from [dqn-atari](../dqn-atari) |
| **PPO** | on-policy actor-critic | ported from [ppo-atari](../ppo-atari) |
| **GRPO** | on-policy, critic-free | the variants from [revisiting-grpo](../revisiting-grpo); *training loop not implemented yet* |

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
│   └── algorithms/
│       ├── __init__.py       # registry: name -> module
│       ├── dqn.py            # QNetwork, flags, training loop
│       ├── ppo.py            # Agent, flags (shared with GRPO), training loop
│       └── grpo.py           # flags and validation; training loop to come
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
python algorithm.py ppo --track --save-model --upload-model --hf-entity byamasupatrick
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
python experiments.py --algos dqn ppo --seeds 1 2 3 --track --save-model
python experiments.py --algos dqn ppo --seeds 1 2 3 --dry-run            # print the commands only
```

- **Shared flags.** Any flag `experiments.py` does not recognise is passed to every run, for example
  `--track`, `--total-timesteps`, or `--save-model`.
- **Flags for one algorithm.** These go in a quoted string, for example
  `--dqn-args "--buffer-size 500000"` or `--ppo-args "--num-envs 16"`.
- **Running locally.** Runs train as parallel processes, at most `--max-parallel` at a time (default:
  the number of CPU cores). Each run writes its output to `logs/{algo}__seed{seed}.log`, and a summary
  prints as runs finish.
- **Algorithms not ready yet.** Unimplemented algorithms (currently `grpo`) are skipped with a
  message, so `--algos dqn ppo grpo` already works.

Size `--max-parallel` to the machine's memory. A DQN run's replay buffer holds 1M stacked frames of
4 × 84 × 84 bytes by default, about 28 GB once full. That much is needed even with
`optimize_memory_usage`, which avoids storing each next observation a second time. PPO needs well
under 1 GB. On a smaller machine, lower it with `--dqn-args "--buffer-size 200000"` (about 5.6 GB).

---

## Running on Hugging Face Jobs

Add `--hf-job` to either command. With `algorithm.py` that submits one run. With `experiments.py`,
every run is submitted as its own Job, so the whole comparison trains in parallel:

```bash
python algorithm.py dqn --track --save-model --upload-model --hf-entity byamasupatrick --hf-job --hf-flavor l40sx1

python experiments.py --algos dqn ppo --seeds 1 2 3 --track --save-model --upload-model \
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
| Evaluation | `src/evaluation.py` | `--eval-episodes` full games on one environment after training, logged as `eval/episodic_return` and `eval/mean_episodic_return`. DQN acts ε-greedily at `--end-e`; PPO samples from its policy. |
| Tracking | `src/run.py` | W&B project `grpo-atari`, run grouped by `--exp-name` and tagged with the algorithm, so runs line up in one workspace. |

### Environment preprocessing

| Wrapper | Flag | Default |
| --- | --- | --- |
| Time limit, counted in agent steps | `--max-episode-steps` | `27000` (108k frames) |
| `RecordEpisodeStatistics` (whole-game, unclipped score) | | always |
| `NoopResetEnv` | `--noop-max` | `30` |
| `MaxAndSkipEnv` | `--frame-skip` | `4` |
| `EpisodicLifeEnv` | `--episodic-life` / `--no-episodic-life` | on |
| `FireResetEnv` | | games with a FIRE action |
| `ClipRewardEnv` | `--clip-rewards` / `--no-clip-rewards` | on |
| `ResizeObservation`, `GrayScaleObservation` | `--screen-size` | `84` |
| `FrameStack` | `--frame-stack` | `4` |

---

## GRPO

GRPO ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)) is PPO with the value network removed.
Each outcome's advantage is its return relative to a group of other samples: the group mean is
subtracted and the result is scaled by the group's standard deviation.
[de Oliveira et al. (2025)](../revisiting-grpo) studied it in classical control. The `grpo`
subcommand exposes their return and baseline variants, and its flags and validation are in place:

| Configuration | Return | Baseline | Critic |
| --- | --- | --- | --- |
| **GRPO (default)** | full-episode Monte Carlo (`--return-type mc --num-steps 0`) | mean of the batch's returns (`--baseline-type batch_mean`), scaled by batch std (`--scale-adv-batch`) | no |
| GRPO, same-seed groups | full-episode Monte Carlo | mean over envs reset with the same seed (`--baseline-type same_seed_mean --num-groups G`) | no |
| TD-n | n-step TD (`--return-type td`) | any `--baseline-type` | optional |
| PPO-equivalent | GAE (`--return-type gae`) | value network | yes |

`same_seed_mean` is the closest analogue of GRPO's group of completions for one prompt. Each group of
environments is reset with a shared seed. `NoFrameskip-v4` games have no sticky actions, so members
of a group diverge only through the policy's sampling and the random no-op starts. The other
baselines (`constant`, `uniform`, `stats`, `ema`) are the revisiting-grpo ablations.

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

Run `pip install` from this directory: `requirements.txt` installs the
[Hugging Face Jobs launcher](../hf-training-jobs) from `../hf-training-jobs`.

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

The same flags as `ppo`, except that `--gae` is absent, `--num-steps` defaults to `0` (full
episodes), and `--norm-adv` defaults to off. It adds:

| Flag | Default | Description |
| --- | --- | --- |
| `--use-value-fn` | off | Train a value head and use it where the configuration needs one. |
| `--return-type` | `mc` | `gae`, `td` (n-step, `n = --num-steps`), or `mc` (full episode). |
| `--baseline-type` | `batch_mean` | `value`, `constant`, `uniform`, `stats`, `batch_mean`, `ema`, or `same_seed_mean`. Ignored for `gae`. |
| `--scale-adv-batch` | on | Divide advantages by the batch standard deviation. |
| `--baseline-constant` | `None` | Baseline for `constant`. |
| `--baseline-uniform-low`, `--baseline-uniform-high` | `None` | Range for `uniform`; falls back to the env's reward range, then `[-1, 1]`. |
| `--baseline-ema-beta` | `0.9` | Decay for `ema`, with Adam-style bias correction. |
| `--num-groups` | `0` | Seed-sharing groups for `same_seed_mean`; must divide `--num-envs`. |

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

## License

MIT, see [LICENSE](../LICENSE) at the repository root. Provenance for the repository as a whole is
recorded in [NOTICE](../NOTICE).
