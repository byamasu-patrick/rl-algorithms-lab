# RL Algorithms Lab

A collection of deep reinforcement learning implementations and experiments, written to be read.
Each subdirectory is a self-contained project with its own dependencies, entry point, and
documentation. The algorithm is never hidden behind a framework layer.

The work builds up to one question: **can a critic-free method, GRPO, learn Atari from pixels, and
how does it compare with DQN and PPO?** To get there, the repository has:

- PPO, taken from classic control up to pixels and continuous control, so one algorithm can be
  compared across problem types;
- DQN, the off-policy value-based counterpart, on classic control and on Atari;
- a research harness asking whether the critic earns its keep in classical environments;
- a project that trains DQN, PPO, and GRPO side by side on Breakout under identical conditions;
- a small package that sends any of these training scripts to Hugging Face Jobs with one flag.

| Project | What it is | Status |
| --- | --- | --- |
| [ppo/](ppo/) | A from-scratch re-implementation of **PPO** for studying the algorithm one detail at a time. Single training loop, every hyperparameter and design choice exposed as a flag. | Working on classic control |
| [ppo-atari/](ppo-atari/) | The same PPO scaled to the **Arcade Learning Environment**: a convolutional policy over stacked frames, plus the standard Atari preprocessing pipeline. | Training runs in progress |
| [ppo-continuous-actions/](ppo-continuous-actions/) | PPO for **continuous control** on the PyBullet robotics tasks: a diagonal Gaussian policy plus observation and reward normalization. | Default task `HalfCheetahBulletEnv-v0` |
| [dqn/](dqn/) | A from-scratch **Deep Q-Network**: replay buffer, target network, and epsilon-greedy exploration. The off-policy, value-based contrast to the PPO directories. | Working on classic control |
| [dqn-atari/](dqn-atari/) | The **Nature DQN** on Atari: the paper's convolutional Q-network, frame preprocessing, evaluation, a Hub upload with a model card, and training on Hugging Face Jobs. | Full 10M-step Breakout run in progress |
| [revisiting-grpo/](revisiting-grpo/) | An experiment harness for asking **whether the critic is necessary**, swapping the learned value baseline for group-statistic alternatives across a large sweep. | Reproduction of published results |
| [grpo-atari/](grpo-atari/) | **DQN vs PPO vs GRPO on Atari**: the three algorithms as interchangeable modules over one shared environment, logging, evaluation, and upload pipeline, runnable alone or as parallel sweeps. | DQN and PPO working; GRPO training loop next |
| [hf-training-jobs/](hf-training-jobs/) | **`hf-jobs-launch`**: add `--hf-job` to any training command and it runs on Hugging Face Jobs instead. Installable from PyPI or GitHub. | Used by `dqn-atari/` and `grpo-atari/`; ready to publish |

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
├── dqn/                    # from-scratch DQN on classic control (pip)
│   ├── algorithm.py        # arguments, Q network, training loop
│   ├── utils.py            # replay buffer
│   ├── eval.py             # greedy evaluation of a checkpoint
│   ├── hugging_face.py     # optional Hub upload
│   └── README.md           # TD target, target network, autoreset notes
│
├── dqn-atari/              # Nature DQN on Atari (pip, Python 3.11)
│   ├── algorithm.py        # arguments, Atari wrappers, training loop
│   ├── src/agent.py        # Nature CNN Q-network and epsilon schedule
│   ├── utils.py            # replay buffer
│   ├── eval.py             # evaluation of a saved checkpoint
│   ├── hugging_face.py     # Hub upload and model card
│   └── README.md           # the paper, preprocessing, defaults vs the paper
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
├── grpo-atari/             # DQN vs PPO vs GRPO on Atari (pip, Python 3.11)
│   ├── algorithm.py        # train one: python algorithm.py {dqn,ppo,grpo}
│   ├── experiments.py      # algorithms x seeds, in parallel, locally or as HF Jobs
│   ├── src/                # shared args, environment, run, evaluation, hub, checkpoints
│   │   └── algorithms/     # dqn.py, ppo.py, grpo.py
│   └── README.md           # layout, comparison protocol, GRPO variants, flags
│
├── hf-training-jobs/       # the hf-jobs-launch package (import: hf_jobs)
│   ├── hf_jobs/            # launcher.py, config.py
│   ├── tests/
│   └── README.md           # install, use, configuration, flags
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

The point is ablation. Each implementation detail that PPO's performance depends on is a
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
loss. The agent sees `(4, 84, 84)` stacked grayscale frames instead of a four-number state vector.
So the network becomes the Nature-DQN convolutional encoder ([Mnih et al.,
2015](https://www.nature.com/articles/nature14236)), with ReLU activations and a trunk **shared**
between actor and critic, where `ppo/` keeps two separate Tanh MLPs.

Raw ALE output is 210x160 RGB at 60 Hz, so eleven composed wrappers reduce it: no-op starts, frame
skipping with max-pooling, life-loss termination, reward clipping, grayscale, resize, and a
4-frame stack. Wrapper *order* matters, and the README works through which orderings are
load-bearing and why.

Evaluation follows [Machado et al., 2017](https://arxiv.org/abs/1709.06009). The `ALE/*-v5`
environment ids default to sticky actions (`repeat_action_probability=0.25`) and a 30-minute
episode cap, which is the main reason to prefer them over the older `NoFrameskip-v4` ids.

```bash
cd ppo-atari
pip install -r requirements.txt
python algorithm.py                      # ALE/Breakout-v5, 10M steps
```

At the defaults this is 10M agent steps across 8 environments. It needs a GPU; on CPU a full run
takes days.

Full documentation is in **[ppo-atari/README.md](ppo-atari/README.md)**: the preprocessing pipeline,
the architecture rationale, how the evaluation protocol maps onto the paper's recommendations, and
the open issues. **Read the known issues before trusting any numbers from this directory.**

### `ppo-continuous-actions/`: from discrete choices to real-valued vectors

Same optimizer again. What changes is the policy distribution. Instead of logits into a
`Categorical`, the actor emits a mean vector, pairs it with a **state-independent learned log-std**,
and samples from a diagonal Gaussian. Log-probabilities and entropy are summed across action
dimensions, since a diagonal Gaussian treats them as independent.

The environment side gains running observation and reward normalization with outlier clipping. That
matters for continuous control, where joint angles and velocities arrive on very different scales.
The entropy bonus is switched off, because the learned log-std shrinking over training already acts
as the exploration schedule.

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
branch of the family. PPO learns a policy from fresh rollouts that are discarded after one update.
DQN instead learns an action-value function from a replay buffer, reuses every transition many
times, and acts by taking the argmax over Q with epsilon-greedy noise on top.

That changes which mechanism keeps training stable. PPO uses a clipped surrogate to stop the policy
moving too far per update. DQN has no policy to constrain. Instead, it holds a delayed copy of the
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

### `dqn-atari/`: the Nature DQN, from pixels

DQN as published in *Human-level control through deep reinforcement learning* ([Mnih et al.,
2015](https://www.nature.com/articles/nature14236)). The network is the paper's three-layer
convolutional Q-network over `4 × 84 × 84` stacked frames. The environment uses the paper's
preprocessing: 30 random no-op starts, a frame skip of 4 with max-pooling, life loss as episode end
during training, reward clipping, grayscale, and an 84×84 resize.
The replay buffer holds 1M transitions in the memory-optimized layout, which stores each frame stack
once.

After training, `--save-model` evaluates the checkpoint for 10 games and records video. With
`--upload-model`, the model is pushed to the Hugging Face Hub with a generated model card that
credits the author, links this repository, and shows how to reproduce and load the model. The
README compares every default with the paper's hyperparameter table.

```bash
cd dqn-atari
py -V:3.11 -m venv .venv && .venv\Scripts\activate        # Python 3.10 or 3.11
pip install -r requirements.txt
python algorithm.py --track --save-model                  # BreakoutNoFrameskip-v4, 10M steps
python algorithm.py --track --save-model --upload-model --hf-job --hf-flavor l40sx1   # on Hugging Face
```

The default budget is 10M agent steps (40M frames). A full run takes about 8 hours on one GPU. The
replay buffer needs about 28 GB of RAM once full, so pick hardware with room for it.

Full documentation is in **[dqn-atari/README.md](dqn-atari/README.md)**.

### `revisiting-grpo/`: is the critic doing the work?

PPO learns a value function to center its advantages. GRPO-style methods drop it, and use statistics
over a *group* of sampled trajectories as the baseline instead. This project asks how much that
costs in classical RL environments, where the critic is cheap and well-conditioned.

The harness makes the baseline a swappable component:

- **Return estimators**: `gae`, `td` (n-step), or `mc` (Monte Carlo), via `--return-type`
- **Baselines**: `value` (a learned critic), `constant`, `uniform`, `stats`, `batch_mean`, `ema`,
  or `same_seed_mean`, via `--baseline-type`
- **Critic on/off**: `--no-use-value-fn` removes the value network and its loss entirely

[experiment.sh](revisiting-grpo/experiment.sh) enumerates the full sweep across three dimensions:
baseline choice (D1), discount and horizon (D2), and group size (D3). Each covers 5 environments
(`CartPole-v1`, `Acrobot-v1`, `MountainCarContinuous-v0`, `HalfCheetah-v4`, `Humanoid-v4`) × 10
seeds. The sweep can be split into shards by instance index across terminals or machines. It skips
runs whose output directory already holds a `config.yaml`, so an interrupted sweep can be resumed.

```bash
cd revisiting-grpo
uv sync && uv sync --extra mujoco
uv run python algorithm.py --no-track        # smoke test on CartPole
bash launch_all_cpus.sh                      # full sweep, all cores
```

Reproduction instructions, the Docker path, and the paper citation are in
**[revisiting-grpo/README.md](revisiting-grpo/README.md)**.

### `grpo-atari/`: DQN vs PPO vs GRPO on Atari

This is where the threads meet. The project trains three algorithms on `BreakoutNoFrameskip-v4`:
DQN (from `dqn-atari/`), PPO (from `ppo-atari/`, moved onto the same Gymnasium 0.29 environment),
and GRPO, which takes the critic-free variants from `revisiting-grpo/` to pixels.

They are compared under identical conditions:

- **Shared modules.** Every algorithm uses the same environment pipeline, step budget, episode
  metrics, evaluation protocol, W&B project, checkpointing, and Hub upload. Differences in the
  results come from the algorithms, not the plumbing.
- **One module per algorithm.** Each lives in `src/algorithms/` with its own flags, checks, and
  training loop. Adding a fourth is one new module and one registry entry.
- **Any combination can run.** `algorithm.py` trains one algorithm. `experiments.py` expands
  algorithms × seeds and runs them as parallel local processes. With `--hf-job`, each run is
  submitted as its own Hugging Face Job, so the whole comparison trains in parallel.

```bash
cd grpo-atari
pip install -r requirements.txt                          # Python 3.10 or 3.11
python algorithm.py dqn --track --save-model             # one algorithm
python algorithm.py ppo --seed 2 --track --save-model
python experiments.py --algos dqn ppo --seeds 1 2 3 --track --save-model            # locally, in parallel
python experiments.py --algos dqn ppo --seeds 1 2 3 --track --save-model --hf-job   # one HF Job per run
```

The `grpo` subcommand already takes its full flag set: return type, baseline type (including
`same_seed_mean` groups, the closest analogue of GRPO's group of completions per prompt), and
advantage scaling. The flags are validated now, and the training loop is the next step.

Full documentation is in **[grpo-atari/README.md](grpo-atari/README.md)**.

### `hf-training-jobs/`: any training script on Hugging Face Jobs

The **`hf-jobs-launch`** package turns a local training command into a
[Hugging Face Job](https://huggingface.co/docs/huggingface_hub/guides/jobs) with one flag:

```python
from hf_jobs import launch

if __name__ == "__main__":
    launch()  # with --hf-job, submits this run to Hugging Face Jobs and exits
    args = parse_args()
```

```bash
python train.py --epochs 10            # runs here
python train.py --epochs 10 --hf-job   # same run, on Hugging Face
```

With `--hf-job`, the launcher:

1. uploads the project to a private bucket;
2. submits a Job that installs the project's dependencies and runs the same command without the
   `--hf-*` flags;
3. passes the Hugging Face token and W&B key as secrets;
4. copies `runs/` and `videos/` back to the bucket when the script exits, even if it fails.

Each project sets its own defaults (namespace, hardware, timeout, image) in `[tool.hf-jobs]` of its
`pyproject.toml`. `--hf-*` flags and `HF_JOBS_*` environment variables override them.

```bash
pip install hf-jobs-launch     # from PyPI, once published
pip install "hf-jobs-launch @ git+https://github.com/byamasu-patrick/rl-algorithms-lab.git#subdirectory=hf-training-jobs"
```

Full documentation is in **[hf-training-jobs/README.md](hf-training-jobs/README.md)**.

---

## Training on Hugging Face Jobs

`dqn-atari/` and `grpo-atari/` call `launch()`, so any of their commands runs remotely by adding
`--hf-job`. Both are configured to run and bill Jobs under the
[baobabtech](https://huggingface.co/baobabtech) organization, on Python 3.11. Models they upload with
`--upload-model` go to the account of whoever submitted the Job, as
`{hf_entity}/{env_id}-{exp_name}-seed{seed}`, with a model card crediting
[Byamasu Patrick Paul](https://huggingface.co/byamasupatrick).

```bash
python algorithm.py ... --hf-job --hf-flavor l40sx1 --hf-timeout 36h   # submit
python algorithm.py ... --hf-job --hf-dry-run                          # show what would be submitted
hf jobs ps --namespace baobabtech                                      # running Jobs
hf jobs logs <job-id> --namespace baobabtech                           # logs
```

Hardware sizing for the Atari runs:

| Run | RAM needed | Suitable flavors |
| --- | --- | --- |
| DQN, 1M replay buffer | about 28 GB once the buffer fills | `l40sx1` (62 GB), `a100-large` (142 GB); `l4x1` (30 GB) runs but with no headroom |
| PPO, 8 environments | well under 1 GB | `t4-small`, `l4x1` |

---

## Why the environments are separate

| | `ppo/` | `ppo-atari/` | `ppo-continuous-actions/` | `dqn/` | `dqn-atari/` | `grpo-atari/` | `revisiting-grpo/` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Installed via | Poetry | Poetry or pip | Poetry | Poetry or pip | pip or Poetry | pip or Poetry | uv |
| Python | 3.10–3.13 | 3.10–3.13 | 3.10–3.13 | 3.10–3.13 | **3.10–3.11** | **3.10–3.11** | 3.10 |
| PyTorch | 2.13.0 | 2.13.0 | 2.13.0 | 2.13.0 | 2.13.0 | 2.13.0 | 2.4.1 |
| Gymnasium | 1.3.0 | 1.3.0 | 1.3.0 | 1.3.0 | **0.29.1** | **0.29.1** | **0.29.1** (plus `gym` 0.23.1) |
| Also needs | | `ale-py` 0.12, `opencv-python` | `pybullet-envs-gymnasium` | `huggingface_hub` | `ale-py` 0.8.1, SB3 2.3.2, `hf-jobs-launch` | `ale-py` 0.8.1, SB3 2.3.2, `hf-jobs-launch` | |

There are two Gymnasium generations here, and they cannot share an environment.

The four classic-control and current-Atari directories track **Gymnasium 1.x**. They could share one
environment, since each only adds its own simulator or tooling on top, but they are kept apart so
each stays installable on its own.

`revisiting-grpo/`, `dqn-atari/`, and `grpo-atari/` are on **Gymnasium 0.29**. The 1.0 release
changed several things that rollout code depends on:

- it rewrote the vector-env autoreset semantics: the terminal observation moved out of
  `info["final_observation"]`, and the reset is now deferred by one step;
- it renamed `FrameStack` and `GrayScaleObservation`;
- it changed the `RecordEpisodeStatistics` info structure.

Rollout and bootstrapping code written against one version does not run correctly against the
other. `revisiting-grpo/` stays pinned to the versions its published results were produced on. The
two Atari comparison projects share that API so the GRPO work can build on it. That pins them to
`ale-py` 0.8.1, the release that registers the `NoFrameskip-v4` games with Gymnasium 0.29. It has no
wheels past Python 3.11, and it needs NumPy 1.26, because Gymnasium 0.29 still uses `np.bool8`.

Create one environment per project, from inside that project's directory. On Windows, the Python
Install Manager (`py install 3.11`, then `py -V:3.11 -m venv .venv`) keeps several Python versions
side by side. The workspace's VS Code settings map `dqn-atari/` and `grpo-atari/` to their own
`.venv`.

---

## Experiment tracking

Every project logs to TensorBoard and can mirror to Weights & Biases with `--track` (run
`wandb login` first).

- **Where runs go:** `dqn-atari/` and `grpo-atari/` log to the `mscsr000324-must` W&B entity, in the
  `dqn-atari` and `grpo-atari` projects. In `grpo-atari`, runs are grouped by experiment name and
  tagged with the algorithm.
- **Run names:** most projects name runs `{env}__{exp_name}__{seed}__{timestamp}`, so runs never
  collide. `grpo-atari/` and `revisiting-grpo/` leave out the timestamp on purpose. There, rerunning
  a finished configuration exits immediately, so an interrupted sweep can be restarted without
  repeating work.
- **What stays local:** `runs/`, `videos/`, and `wandb/` are gitignored repository-wide, so
  experiment output stays local. Runs on Hugging Face Jobs copy them back to the
  `baobabtech/hf-training-jobs` bucket.

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

`dqn-atari/` follows the training loop of CleanRL's `dqn_atari.py` and vendors the same
Stable-Baselines3-derived replay buffer. `grpo-atari/` reuses that buffer, ports the DQN and PPO
loops above, and adapts the argument set and checkpoint utilities of `revisiting-grpo/`.
`hf-training-jobs/` is original work.

## License

MIT, see [LICENSE](LICENSE). The `hf-jobs-launch` package carries its own copy in
[hf-training-jobs/LICENSE](hf-training-jobs/LICENSE).

The three PPO directories, `dqn/`, and `hf-training-jobs/` are original work, aside from the
vendored replay buffer noted above. `revisiting-grpo/`, `dqn-atari/`, and `grpo-atari/` include code
derived from third-party MIT-licensed projects. Copyright notices are retained in
[NOTICE](NOTICE). All directories carry the same terms.
