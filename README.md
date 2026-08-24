# RL Algorithms Lab

A collection of deep reinforcement learning implementations and experiments, written to be read.
Each subdirectory is a self-contained project with its own dependencies, entry point, and
documentation. There is no shared framework layer to trace through, and no abstraction that hides
the algorithm from you.

Three of the directories are PPO, taken from classic control up to pixels and continuous control, so
the same algorithm can be compared across problem types. One is DQN, the off-policy value-based
counterpart. One is a research harness asking whether the critic earns its keep.

| Project | What it is | Status |
| --- | --- | --- |
| [ppo/](ppo/) | A from-scratch re-implementation of **PPO** for studying the algorithm one detail at a time. Single training loop, every hyperparameter and design choice exposed as a flag. | Working on classic control |
| [ppo-atari/](ppo-atari/) | The same PPO algorithm scaled to the **Arcade Learning Environment**: a convolutional policy over stacked frames, plus the standard Atari preprocessing pipeline. | Training runs in progress |
| [ppo-continuous-actions/](ppo-continuous-actions/) | PPO for **continuous control** on the PyBullet robotics tasks: a diagonal Gaussian policy plus observation and reward normalization. | Default task `HalfCheetahBulletEnv-v0` |
| [dqn/](dqn/) | A from-scratch **Deep Q-Network**: replay buffer, target network, and epsilon-greedy exploration. The off-policy, value-based contrast to the PPO directories. | Working on classic control |
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
├── ppo-continuous-actions/ # PPO for continuous control on PyBullet (pip)
│   ├── algorithm.py        # training loop + normalization wrappers
│   ├── src/agent.py        # Gaussian policy with a learned log-std
│   └── README.md           # Gaussian policy, normalization, hyperparameters
│
├── dqn/                    # from-scratch DQN (pip)
│   ├── algorithm.py        # arguments, Q network, training loop
│   ├── utils.py            # replay buffer
│   ├── eval.py             # greedy evaluation of a checkpoint
│   ├── hugging_face.py     # optional Hub upload
│   └── README.md           # TD target, target network, autoreset notes
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

## The projects

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
poetry install
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

### `ppo-continuous-actions/`: from discrete choices to real-valued vectors

Same optimizer again. What changes is the policy distribution: instead of logits into a
`Categorical`, the actor emits a mean vector, pairs it with a **state-independent learned log-std**,
and samples from a diagonal Gaussian. Log-probabilities and entropy are summed across action
dimensions, since a diagonal Gaussian treats them as independent.

The environment side gains running observation and reward normalization with outlier clipping, which
continuous control needs because joint angles and velocities arrive on very different scales. The
entropy bonus is switched off, because the learned log-std shrinking over training is already the
exploration schedule.

Tasks come from PyBullet rather than MuJoCo, so no licence is required. The trade-off is that scores
are not comparable to MuJoCo `HalfCheetah-v4`.

```bash
cd ppo-continuous-actions
pip install -r requirements.txt
pip install pybullet-envs-gymnasium          # not listed in either dependency file
python algorithm.py                          # HalfCheetahBulletEnv-v0, 2M steps
```

Full documentation is in
**[ppo-continuous-actions/README.md](ppo-continuous-actions/README.md)**, including the two open
issues that stop a fresh clone from running.

### `dqn/`: the off-policy, value-based contrast

The three directories above are all the same on-policy policy-gradient method. This one is the other
branch of the family. Instead of learning a policy from fresh rollouts that are thrown away after
one update, DQN learns an action-value function from a replay buffer, reuses every transition many
times, and acts by taking the argmax over Q with epsilon-greedy noise on top.

That swaps out which machinery does the stabilizing. PPO uses a clipped surrogate to stop the policy
moving too far per update; DQN has no policy to constrain, and instead holds a delayed copy of the
network as the regression target, so the network is not chasing its own output.

Also included are a greedy evaluation pass over a saved checkpoint and an optional Hugging Face Hub
upload with a generated model card.

```bash
cd dqn
pip install -r requirements.txt
python algorithm.py                          # CartPole-v1, 500k steps
```

Full documentation is in **[dqn/README.md](dqn/README.md)**: the TD target, why the target network is
delayed, the full flag reference, and why the vector environments opt out of Gymnasium's default
autoreset mode.

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

| | `ppo/` | `ppo-atari/` | `ppo-continuous-actions/` | `dqn/` | `revisiting-grpo/` |
| --- | --- | --- | --- | --- | --- |
| Installed via | Poetry | Poetry or pip | Poetry | Poetry or pip | uv |
| PyTorch | 2.13.0 | 2.13.0 | 2.13.0 | 2.13.0 | 2.4.1 |
| Gymnasium | 1.3.0 | 1.3.0 | 1.3.0 | 1.3.0 | 0.29.1 (plus `gym` 0.23.1) |
| Also needs | | `ale-py`, `opencv-python` | `pybullet-envs-gymnasium` | `huggingface_hub` | |

The four current-Gymnasium directories pin the same PyTorch and Gymnasium, so they *could* share one
environment; each only adds its own simulator or tooling layer on top. They are kept apart so each
stays installable on its own, not because they conflict.

Each of those four declares its dependencies in its own `pyproject.toml` with
`package-mode = false`, so `poetry install` resolves dependencies without trying to build a package.
`ppo-continuous-actions/` is Poetry-only: its `requirements.txt` was frozen without PyTorch and
cannot produce a runnable environment on its own.

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

All five projects log to TensorBoard by default and can mirror to Weights & Biases with `--track`
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

The three PPO directories and `dqn/` are independent from-scratch implementations, written for
study, and follow CleanRL's single-file structure by convention. `ppo-atari/` reuses the Atari
wrappers from [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) as an installed
dependency rather than vendored source, and its network follows the architecture published in Mnih
et al. (2015). `dqn/` vendors a replay buffer derived from Stable-Baselines3 into
[dqn/utils.py](dqn/utils.py), which is MIT-licensed; see [NOTICE](NOTICE).

## License

MIT, see [LICENSE](LICENSE).

The three PPO directories and `dqn/` are original work, aside from the vendored replay buffer noted
above. `revisiting-grpo/` is derived from third-party MIT-licensed projects. Copyright notices for
both are retained in [NOTICE](NOTICE). All directories carry the same terms.
