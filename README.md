# RL Algorithms Lab

A collection of deep reinforcement learning implementations and experiments, written to be read.
Each subdirectory is a self-contained project with its own dependencies, entry point, and
documentation. There is no shared framework layer to trace through, and no abstraction that hides
the algorithm from you.

| Project | What it is | Status |
| --- | --- | --- |
| [ppo/](ppo/) | A from-scratch re-implementation of **PPO** for studying the algorithm one detail at a time. Single training loop, every hyperparameter and design choice exposed as a flag. | Working on classic control |
| [revisiting-grpo/](revisiting-grpo/) | An experiment harness for asking **whether the critic is necessary**, swapping the learned value baseline for group-statistic alternatives across a large sweep. | Reproduction of published results |

---

## Repository layout

```text
.
├── ppo/                    # from-scratch PPO (Poetry)
│   ├── algorithm.py        # argument parsing + full training loop
│   ├── src/agent.py        # actor/critic networks
│   └── README.md           # algorithm walkthrough, CLI reference, metric guide
│
├── revisiting-grpo/        # critic-free baseline study (uv + Docker)
│   ├── algorithm.py        # entry point
│   ├── environment.py      # environment construction and wrappers
│   ├── src/                # args, rollout, returns, training, checkpoints
│   ├── experiment.sh       # enumerates and shards the full experiment sweep
│   ├── launch_all_cpus.sh  # runs the sweep across all available cores
│   ├── Dockerfile
│   └── README.md           # reproduction instructions and citation
│
└── .gitignore              # shared: runs/, videos/, wandb/, __pycache__/
```

---

## The two projects

### `ppo/`: Proximal Policy Optimization, from scratch

A single readable training loop implementing PPO ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)):
clipped surrogate objective, generalized advantage estimation, vectorized rollout collection, and
shuffled minibatch updates. Written in the CleanRL single-file style, so the algorithm lives in one
file and the control flow is visible end to end.

The point is ablation. Every implementation detail that PPO's performance actually depends on is a
command-line flag: advantage normalization, value-loss clipping, learning-rate annealing, the
entropy bonus, gradient clipping, and GAE vs. Monte-Carlo returns. Flip one, rerun, compare.

Currently supports **discrete action spaces** on classic control environments.

```bash
cd ppo
poetry install --no-root
python algorithm.py                    # CartPole-v1, 25k steps
tensorboard --logdir runs
```

Full documentation is in **[ppo/README.md](ppo/README.md)**: an algorithm walkthrough with the
equations mapped to line numbers, all 22 CLI flags, how to read each logged metric, and the known
deviations from the reference implementation.

### `revisiting-grpo/`: is the critic doing the work?

PPO learns a value function to center its advantages. GRPO-style methods drop it, using statistics
over a *group* of sampled trajectories as the baseline instead. This project asks how much that
actually costs in classical RL environments, where the critic is cheap and well-conditioned.

The harness makes the baseline a swappable component:

- **Return estimators**: `gae`, `td` (n-step), or `mc` (Monte Carlo), via `--return-type`
- **Baselines**: `value` (a learned critic), `constant`, `uniform`, `stats`, `batch_mean`, `ema`,
  or `same_seed_mean`, via `--baseline-type`
- **Critic on/off**: `--no-use-value-fn` removes the value network and its loss entirely

[experiment.sh](revisiting-grpo/experiment.sh) enumerates the full sweep across three dimensions:
baseline choice (D1), discount and horizon (D2), and group size (D3). Each covers 5 environments
(`CartPole-v1`, `Acrobot-v1`, `MountainCarContinuous-v0`, `HalfCheetah-v4`, `Humanoid-v4`) × 10
seeds. It shards by instance index so the sweep can be split across terminals or machines, and skips
runs whose output directory already holds a `config.yaml`, making it resumable.

```bash
cd revisiting-grpo
uv sync && uv sync --extra mujoco
uv run python algorithm.py --no-track        # smoke test on CartPole
bash launch_all_cpus.sh                      # full sweep, all cores
```

Reproduction instructions, the Docker path, and the paper citation are in
**[revisiting-grpo/README.md](revisiting-grpo/README.md)**.

---

## Why the environments are separate

The two projects **cannot share a virtual environment**, and the split is deliberate rather than
untidy:

| | `ppo/` | `revisiting-grpo/` |
| --- | --- | --- |
| Package manager | Poetry | uv |
| Python | `>=3.10, <3.14` | `>=3.10, <3.11` |
| PyTorch | 2.13.0 | 2.4.1 |
| Gymnasium | 1.3.0 | 0.29.1 (plus `gym` 0.23.1) |

The Gymnasium versions straddle the 1.0 release, which rewrote the vector-env autoreset semantics
(the terminal observation moved out of `info["final_observation"]`, and the reset is now deferred by
one step) and reorganised the wrappers and the `RecordEpisodeStatistics` info structure. Rollout and
bootstrapping code written against one version does not run correctly against the other. `ppo/`
tracks current Gymnasium; `revisiting-grpo/` stays pinned to the versions the published results were
produced on, because moving it would invalidate the comparison.

Create one environment per project, from inside that project's directory.

---

## Experiment tracking

Both projects log to TensorBoard by default and can mirror to Weights & Biases with `--track`
(export `WANDB_API_KEY`, or run `wandb login`, first). Run directories are named
`{env}__{exp_name}__{seed}__{timestamp}` so runs never collide.

`runs/`, `videos/`, and `wandb/` are gitignored repository-wide, so experiment output stays local.

---

## Attribution

`revisiting-grpo/` builds on the code released with *Learning Without Critics? Revisiting GRPO in
Classical Reinforcement Learning Environments* (de Oliveira et al., Latinx in AI @ NeurIPS 2025),
which itself derives from [CleanRL](https://github.com/vwxyzjn/cleanrl). If you use that work,
please cite the paper. The BibTeX entry is in
[revisiting-grpo/README.md](revisiting-grpo/README.md#citing).

`ppo/` is an independent from-scratch implementation, written for study, and follows CleanRL's
single-file structure by convention.

## License

MIT.
