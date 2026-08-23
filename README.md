# RL Algorithms Lab

A collection of deep reinforcement learning implementations and experiments, written to be read.
Each subdirectory is a self-contained project with its own dependencies, entry point, and
documentation. There is no shared framework layer to trace through, and no abstraction that hides
the algorithm from you.

| Project | What it is | Status |
| --- | --- | --- |
| [ppo/](ppo/) | A from-scratch re-implementation of **PPO** for studying the algorithm one detail at a time. Single training loop, every hyperparameter and design choice exposed as a flag. | Working on classic control |
| [ppo-atari/](ppo-atari/) | The same PPO algorithm scaled to the **Arcade Learning Environment**: a convolutional policy over stacked frames, plus the standard Atari preprocessing pipeline. | Training runs in progress |
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
├── ppo-atari/              # PPO on the Arcade Learning Environment (pip)
│   ├── algorithm.py        # training loop + the Atari preprocessing stack
│   ├── src/agent.py        # Nature-DQN CNN, shared trunk with two heads
│   └── README.md           # preprocessing, CNN rationale, evaluation protocol
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
├── LICENSE                 # MIT
├── NOTICE                  # provenance and third-party copyright notices
└── .gitignore              # shared: runs/, videos/, wandb/, .venv/, __pycache__/
```

---

## The three projects

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

### `ppo-atari/`: the same algorithm, a much harder perception problem

The optimization code here is byte-identical to `ppo/`. Everything that changes is upstream of the
loss: the agent sees `(4, 84, 84)` stacked grayscale frames instead of a four-number state vector,
so the network becomes the Nature-DQN convolutional encoder ([Mnih et al.,
2015](https://www.nature.com/articles/nature14236)) with ReLU activations and a trunk **shared**
between actor and critic, where `ppo/` keeps two separate Tanh MLPs.

Raw ALE output is 210x160 RGB at 60 Hz, so eleven composed wrappers reduce it: no-op starts, frame
skipping with max-pooling, life-loss termination, reward clipping, grayscale, resize, and a
4-frame stack. Wrapper *order* carries real weight, and the README works through which orderings
are load-bearing and why.

Evaluation follows [Machado et al., 2017](https://arxiv.org/abs/1709.06009). The `ALE/*-v5`
environment ids default to sticky actions (`repeat_action_probability=0.25`) and a 30-minute
episode cap, which is the main reason to prefer them over the older `NoFrameskip-v4` ids.

```bash
cd ppo-atari
pip install -r requirements.txt
python algorithm.py                      # ALE/Breakout-v5, 10M steps
```

At the defaults this is 10M agent steps across 8 environments. It wants a GPU; on CPU a full run
takes days.

Full documentation is in **[ppo-atari/README.md](ppo-atari/README.md)**: the preprocessing pipeline,
the architecture rationale, how the evaluation protocol maps onto the paper's recommendations, and
the open issues. **Read the known issues before trusting any numbers from this directory.**

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

| | `ppo/` | `ppo-atari/` | `revisiting-grpo/` |
| --- | --- | --- | --- |
| Installed via | Poetry | pip, `requirements.txt` | uv |
| Python | `>=3.10, <3.14` | 3.12 as created | `>=3.10, <3.11` |
| PyTorch | 2.13.0 | 2.13.0 | 2.4.1 |
| Gymnasium | 1.3.0 | 1.3.0 | 0.29.1 (plus `gym` 0.23.1) |
| Also needs | | `ale-py` 0.12.1, `opencv-python` | |

`ppo/` and `ppo-atari/` pin the same PyTorch and Gymnasium, so they *could* share one environment;
`ppo-atari/` only adds the ALE and OpenCV layers on top. They are kept apart so each stays
installable on its own, not because they conflict.

`revisiting-grpo/` genuinely cannot join them. Its Gymnasium pin sits on the far side of the 1.0
release, which rewrote the vector-env autoreset semantics (the terminal observation moved out of
`info["final_observation"]`, and the reset is now deferred by one step) and reorganised the wrappers
and the `RecordEpisodeStatistics` info structure. Rollout and bootstrapping code written against one
version does not run correctly against the other. The two PPO directories track current Gymnasium;
`revisiting-grpo/` stays pinned to the versions the published results were produced on, because
moving it would invalidate the comparison.

Create one environment per project, from inside that project's directory.

---

## Experiment tracking

All three projects log to TensorBoard by default and can mirror to Weights & Biases with `--track`
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

`ppo/` and `ppo-atari/` are independent from-scratch implementations, written for study, and follow
CleanRL's single-file structure by convention. `ppo-atari/` reuses the Atari wrappers from
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) as an installed dependency rather
than vendored source, and its network follows the architecture published in Mnih et al. (2015).

## License

MIT, see [LICENSE](LICENSE).

`ppo/` and `ppo-atari/` are original work. `revisiting-grpo/` is derived from third-party
MIT-licensed projects, whose copyright notices are retained in [NOTICE](NOTICE). All three
directories carry the same terms.
