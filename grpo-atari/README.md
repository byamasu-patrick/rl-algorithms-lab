# GRPO on Atari

Can **Group Relative Policy Optimization** learn to play Atari from pixels without a critic?

GRPO ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)) is PPO with the value network removed.
Instead of a learned baseline, each sampled outcome's advantage is its return relative to other
samples of the same prompt: the group's mean is subtracted and the result is divided by the group's
standard deviation. [de Oliveira et al. (2025)](../revisiting-grpo) took the idea back to classical
control and MuJoCo. This project asks the same question on Atari, with the same return and baseline
variants as [revisiting-grpo](../revisiting-grpo), stacked-frame observations, and the standard
Atari preprocessing from [dqn-atari](../dqn-atari).

The default game is `BreakoutNoFrameskip-v4`. Training runs locally or, with one extra flag, on
[Hugging Face Jobs](#training-on-hugging-face-jobs).

> **Status: project set up, algorithm not yet implemented.** Arguments, environment preprocessing,
> seeding, logging, W&B tracking, and the Jobs integration are in place. `algorithm.py` builds the
> environments, writes the run configuration, and stops where the agent and training loop will go.

---

## From PPO to GRPO

Every variant is PPO's clipped surrogate objective; they differ only in how the advantage `A_t` is
estimated. The flags select the two parts separately: the **return** each step is credited with,
and the **baseline** subtracted from it.

| Configuration | Return | Baseline | Critic |
| --- | --- | --- | --- |
| PPO | GAE over `--num-steps` (`--return-type gae`) | value network | yes |
| PPO, TD-n | n-step TD (`--return-type td`) | any `--baseline-type` | optional |
| **GRPO (default)** | full-episode Monte Carlo (`--return-type mc --num-steps 0`) | mean of the batch's returns (`--baseline-type batch_mean`), scaled by batch std (`--scale-adv-batch`) | no |
| GRPO, same-seed groups | full-episode Monte Carlo | mean over envs reset with the same seed (`--baseline-type same_seed_mean --num-groups G`) | no |

`same_seed_mean` is the closest analogue of GRPO's group of completions for one prompt. The
`--num-envs` environments are split into `--num-groups` groups. Each group is reset with a shared
seed, so its members start from the same state, and each env's return is compared with its group's
mean. `NoFrameskip-v4` games have no sticky actions, so within a group the only divergence comes
from the policy's own sampling and the random no-op starts.

The other baselines (`constant`, `uniform`, `stats`, `ema`) are the ablations from
revisiting-grpo, kept so the same sweeps can be run on Atari.

---

## Environment preprocessing

[src/environment.py](src/environment.py) applies the standard Atari stack. Each step is controlled
by a flag so its effect on a critic-free method can be ablated:

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

Two of these matter more for GRPO than for PPO. With `--num-steps 0` an "episode" is what the
rollout waits for, so `--episodic-life` (one life per episode) and `--max-episode-steps` set how
long each rollout takes and what a Monte Carlo return covers.

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

The dependency set is the one from revisiting-grpo, minus what this project does not use (`gym`,
`tyro`, `rich`, the MuJoCo and JAX extras), plus the Atari stack from dqn-atari:

| Pin | Reason |
| --- | --- |
| `gymnasium[atari,accept-rom-license]==0.29.1` | Same Gymnasium as revisiting-grpo; the extras install `shimmy`, `ale-py`, and the ROMs. |
| `ale-py==0.8.1` | Registers the `*NoFrameskip-v4` environments with Gymnasium 0.29. Limits Python to ≤ 3.11. |
| `numpy==1.26.4` | Gymnasium 0.29's environment checker uses `np.bool8`, which NumPy 2 removed. |
| `stable-baselines3==2.3.2` | Atari wrappers; last line compatible with Gymnasium 0.29. |
| `opencv-python-headless` | Needed by `ResizeObservation`, without the `libGL` dependency missing from the Jobs image. |
| `moviepy==1.0.3` | Gymnasium 0.29's `RecordVideo` is written against the MoviePy 1.x API. |
| `PyYAML`, `tqdm` | `config.yaml` in each run directory, and the progress bar. |
| `torch==2.13.0` | Same as dqn-atari. The Windows pip wheel is CPU-only; the Linux wheels used in Jobs include CUDA. |

---

## Running

Run from this directory.

```bash
python algorithm.py                                              # GRPO, batch-mean baseline
python algorithm.py --baseline-type same_seed_mean --num-groups 4
python algorithm.py --return-type gae --num-steps 128 --use-value-fn --no-scale-adv-batch   # PPO
python algorithm.py --track --save-model --upload-model --hf-entity byamasupatrick
tensorboard --logdir runs
```

Each run writes to `runs/{run_name}/`, where `run_name` defaults to
`{exp_name}__{env_id}__{seed}` (for example `algorithm__breakoutnoframeskip_v4__1`), and saves
the full configuration there as `config.yaml`. With no timestamp in the name, rerunning the same
configuration exits early instead of overwriting; pass `-o` / `--overwrite` to replace it. This
matches revisiting-grpo, where sweeps skip runs that already finished.

Invalid flag combinations stop the run before anything is created, for example
`--return-type td` without `--num-steps > 0`, or `--baseline-type same_seed_mean` without a
`--num-groups` that divides `--num-envs`.

---

## Training on Hugging Face Jobs

Add `--hf-job` to any command to run it on Hugging Face instead of this machine:

```bash
python algorithm.py --track --save-model --upload-model --hf-entity byamasupatrick --hf-job --hf-flavor l4x1
```

The Job runs and is billed under the `baobabtech` organization, logs to your W&B account, and
uploads the model to your own Hugging Face account. The only hook in the code is the first line of
the `__main__` block in [algorithm.py](algorithm.py):

```python
launch()  # with --hf-job, submits this run to Hugging Face Jobs and exits
```

Check what would be submitted with `--hf-dry-run`. See [hf-training-jobs](../hf-training-jobs) for
the other `--hf-*` flags, where outputs are stored, and how to monitor a Job.

---

## Command-line reference

Boolean flags take a `--no-` form to turn them off, for example `--no-episodic-life`.

### Experiment setup

| Flag | Default | Description |
| --- | --- | --- |
| `--exp-name` | file name (`algorithm`) | Name of this experiment; also the W&B group. |
| `--run-name` | `{exp_name}__{env_id}__{seed}` | Run directory and W&B run name. |
| `--seed` | `1` | Seeds Python, NumPy, Torch, and the env action spaces. |
| `--torch-deterministic` | on | Sets `torch.backends.cudnn.deterministic`. |
| `--cuda` | on | Use CUDA when available. |
| `--track` | off | Mirror metrics to Weights & Biases. |
| `--wandb-project-name` | `grpo-atari` | W&B project. |
| `--wandb-entity` | `None` | W&B team/entity; `None` uses your default entity. |
| `--capture-video` | off | Record episodes from the first environment to `videos/{run_name}`. |
| `--num-checkpoints` | `10` | Checkpoints saved over the run; sets `--checkpoint-every`. `0` disables. |
| `--checkpoint-every` | `0` | Env steps between checkpoints, when `--num-checkpoints` is `0`. |
| `--checkpoint-load-path` | `None` | Checkpoint to resume from. |
| `--checkpoint-param-filters` | `None` | JSON filter selecting which checkpoint parameters to load. |
| `--log-every` | `0` | Env steps between detailed stat logs; `0` logs every iteration. |
| `-o`, `--overwrite` | off | Replace an existing run directory instead of exiting. |
| `--save-model` | off | Save the final model to `runs/{run_name}`. |
| `--upload-model` | off | Push the saved model to the Hugging Face Hub. Requires `--save-model`. |
| `--hf-entity` | `""` | Hub user or org for the model repository; empty means the token's account. |

### Environment

| Flag | Default | Description |
| --- | --- | --- |
| `--env-id` | `BreakoutNoFrameskip-v4` | Gymnasium Atari id; use the `NoFrameskip-v4` variants. |
| `--env-configs` | `None` | JSON dict of extra `gym.make` keyword arguments. |
| `--noop-max` | `30` | Maximum random no-ops at reset; `0` disables. |
| `--frame-skip` | `4` | Frames each action is repeated for. |
| `--frame-stack` | `4` | Frames stacked as the observation. |
| `--screen-size` | `84` | Height and width of the resized grayscale frame. |
| `--episodic-life` | on | End the training episode when a life is lost. |
| `--clip-rewards` | on | Clip rewards to `{-1, 0, +1}`. |
| `--max-episode-steps` | `27000` | Truncate episodes after this many agent steps; `0` disables. |

### Algorithm

| Flag | Default | Description |
| --- | --- | --- |
| `--total-timesteps` | `10000000` | Total environment steps across all envs. |
| `--learning-rate` | `2.5e-4` | Adam learning rate. |
| `--num-envs` | `8` | Parallel environments; with MC returns, the episodes per update. |
| `--num-steps` | `0` | Steps per env per rollout; `0` collects one full episode per env. |
| `--anneal-lr` | on | Linearly anneal the learning rate to 0. |
| `--gamma` | `0.99` | Discount factor. |
| `--gae-lambda` | `0.95` | GAE lambda, with `--return-type gae`. |
| `--num-minibatches` | `4` | Minibatches per epoch. |
| `--update-epochs` | `4` | Epochs over each batch. |
| `--norm-adv` | off | Normalize advantages within each minibatch. |
| `--clip-coef` | `0.1` | Surrogate clipping coefficient (the Atari PPO value). |
| `--clip-vloss` | on | Clip the value loss, when a value function is used. |
| `--ent-coef` | `0.01` | Entropy bonus coefficient. |
| `--vf-coef` | `0.5` | Value loss coefficient. |
| `--max-grad-norm` | `0.5` | Gradient clipping norm. |
| `--target-kl` | `None` | Stop the epoch loop early above this approximate KL. |

### Returns and baselines

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

`--batch-size`, `--minibatch-size`, and `--num-iterations` are computed from the others and
recorded in `config.yaml`. With `--num-steps 0` the batch size varies per iteration and training
stops on `--total-timesteps`.

---

## Layout

```text
grpo-atari/
├── algorithm.py         # entry point: setup, and (next) the training loop
├── agent.py             # policy network (to be implemented)
├── src/
│   ├── args.py          # all flags, and validation of flag combinations
│   └── environment.py   # Atari environment factory
├── pyproject.toml
└── requirements.txt     # pip-installable pins, including the Jobs launcher
```

---

## References

- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*
  (2024), which introduced GRPO. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- de Oliveira et al., *Learning Without Critics? Revisiting GRPO in Classical Reinforcement Learning
  Environments*. Latinx in AI @ NeurIPS 2025. Code and citation in [revisiting-grpo](../revisiting-grpo).
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017).
  [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Mnih et al., *Human-level control through deep reinforcement learning* (2015), for the Atari
  preprocessing. [Nature 518, 529–533](https://www.nature.com/articles/nature14236)

## License

MIT, see [LICENSE](../LICENSE) at the repository root. Provenance for the repository as a whole is
recorded in [NOTICE](../NOTICE).
