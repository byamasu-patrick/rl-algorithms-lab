# DQN on Atari

A from-scratch implementation of the **Deep Q-Network** from Mnih et al.,
[*Human-level control through deep reinforcement learning*](https://www.nature.com/articles/nature14236),
Nature 518, 529–533 (2015). The agent learns an action-value function `Q(s, a)` directly from
stacked screen frames, off-policy from a replay buffer, and acts by taking the argmax over it.

This is the Atari counterpart to the PPO projects in this repository. The default game is
`BreakoutNoFrameskip-v4`.

> **Status: work in progress.** The argument parser, environment setup, and the paper's Q-network
> are in place. The Atari preprocessing wrappers, replay buffer, and training loop are not written
> yet. See [What is implemented](#what-is-implemented).

---

## The paper in brief

The Nature paper trained one network architecture, with one set of hyperparameters, on 49 Atari
2600 games, seeing only raw pixels, the score, and the set of legal actions. Two changes to plain
Q-learning with a neural network made this stable:

1. **Experience replay.** Transitions `(s, a, r, s')` are stored in a large memory and learned from in
   uniformly sampled minibatches. This breaks the correlation between consecutive samples and reuses
   each transition many times.
2. **A separate target network.** The bootstrap target is computed with a copy of the network whose
   weights `θ⁻` are only refreshed every `C` updates, so the network is not regressing onto a target
   that moves with every gradient step.

The loss at iteration `i` is

$$
L_i(\theta_i) = \mathbb{E}_{(s,a,r,s') \sim U(D)} \Big[ \big( r + \gamma \max_{a'} Q(s', a'; \theta_i^-) - Q(s, a; \theta_i) \big)^2 \Big]
$$

with the TD error clipped to `[-1, 1]` and rewards clipped to `[-1, 1]` during training.

### Preprocessing

From the paper's Methods section:

- take the pixel-wise max over the current and previous frame to remove sprite flicker,
- convert to luminance and rescale to `84 × 84`,
- repeat each chosen action for 4 frames (frame skip),
- stack the last 4 preprocessed frames as the network input,
- start each evaluation episode with up to 30 random no-op actions.

During training, losing a life is treated as the end of an episode in games that have a life
counter.

### Network

[src/agent.py](src/agent.py) implements the paper's architecture:

```text
input   4 × 84 × 84   (stacked frames, scaled by 1/255)
conv1   32 filters, 8×8, stride 4, ReLU   -> 32 × 20 × 20
conv2   64 filters, 4×4, stride 2, ReLU   -> 64 × 9 × 9
conv3   64 filters, 3×3, stride 1, ReLU   -> 64 × 7 × 7
fc      3136 -> 512, ReLU
out     512 -> n_actions  (one Q value per action)
```

A single forward pass scores every action, which is why the output layer has one unit per action
rather than taking the action as input.

---

## What is implemented

| Piece | State |
| --- | --- |
| Command-line arguments | Done, see [Command-line reference](#command-line-reference). |
| Seeding, TensorBoard writer, optional W&B tracking | Done. |
| Vector environment with episode statistics and optional video | Done. |
| Q-network ([src/agent.py](src/agent.py)) | Done, matches the paper. Not yet imported by `algorithm.py`. |
| Atari environments | **Missing:** `ale-py` is not a dependency yet, and `gym.register_envs(ale_py)` is not called, so `BreakoutNoFrameskip-v4` will not load. |
| Preprocessing wrappers (no-op reset, frame skip + max, episodic life, reward clipping, grayscale, 84×84 resize, frame stack) | **Missing.** Without them the observations will not match the network's `4 × 84 × 84` input. |
| Replay buffer | **Missing.** |
| Epsilon schedule, TD update, target network sync | **Missing.** |
| Checkpoint saving and Hugging Face upload (`--save-model`, `--upload-model`) | Flags exist; the code behind them does not. |

---

## Installation

The project runs on Python 3.10–3.13 (`requires-python = ">=3.10,<3.14"`). The local `.venv` is
built on **Python 3.12**.

On Windows, Python versions are managed with the official Python Install Manager (`py`):

```powershell
py list                       # installed versions
py install 3.12               # add a version
py -V:3.12 -m venv .venv      # create the venv on 3.12
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with Poetry:

```bash
poetry install
poetry install --extras hub   # adds huggingface-hub and tenacity for the upload path
```

Core pins are `torch` 2.13.0, `gymnasium` 1.3.0, and `numpy` 2.5.2.

Notes:

- **pygame-ce instead of pygame.** Gymnasium needs pygame to render the classic-control
  environments. `pygame==2.6.1` has no wheels for Python 3.14 and fails to build from source, so the
  dependency is `pygame-ce`, a maintained drop-in fork that is still imported as `import pygame`.
- **`distutils`.** `algorithm.py` imports `strtobool` from `distutils.util`. `distutils` was removed
  from the standard library in Python 3.12; the import works because `setuptools` (pinned in
  `requirements.txt`) provides a replacement. Keep `setuptools` installed in the environment.
- **CPU-only PyTorch.** A plain `pip install` on Windows gets the CPU build of torch. To train on an
  NVIDIA GPU, reinstall torch from the CUDA index listed on the PyTorch site.
- **VS Code.** The workspace settings point the Python extension at `dqn-atari/.venv`. If the status
  bar still shows another interpreter, run *Python: Select Interpreter* and pick the `.venv` entry.

---

## Running

Run from this directory:

```bash
python algorithm.py
python algorithm.py --env-id PongNoFrameskip-v4 --seed 7
tensorboard --logdir runs
```

Runs are written to `runs/{env_id}__{exp_name}__{seed}__{timestamp}/`, and every flag is recorded in
the TensorBoard `hyperparameters` text panel.

Until the pieces listed as missing above are written, `algorithm.py` builds the environments,
prints the action space, and exits.

---

## Command-line reference

Boolean flags accept a bare flag (`--track`) or an explicit value (`--track false`).

### Experiment setup

| Flag | Default | Description |
| --- | --- | --- |
| `--exp-name` | file name (`algorithm`) | Name of this experiment; part of the run directory name. |
| `--seed` | `1` | Seeds Python, NumPy, Torch, and the env spaces. |
| `--torch-deterministic` | `True` | Sets `torch.backends.cudnn.deterministic`. |
| `--cuda` | `True` | Use CUDA when available. |
| `--track` | `False` | Mirror metrics to Weights & Biases. |
| `--wandb-project-name` | `cleanRL` | W&B project. |
| `--wandb-entity` | `None` | W&B team/entity. |
| `--capture-video` | `False` | Record episodes from the first environment. |
| `--save-model` | `False` | Save the model to `runs/{run_name}`. |
| `--upload-model` | `False` | Upload the saved model to the Hugging Face Hub. |
| `--hf-entity` | `""` | Hub user or org for the model repository. |

### Algorithm

| Flag | Default | Description |
| --- | --- | --- |
| `--env-id` | `BreakoutNoFrameskip-v4` | Gymnasium environment id. Must have a **discrete** action space. |
| `--total-timesteps` | `10000000` | Total environment steps. |
| `--learning-rate` | `1e-4` | Optimizer learning rate. |
| `--num-envs` | `1` | Parallel environments. |
| `--buffer-size` | `1000000` | Replay memory capacity in transitions. |
| `--gamma` | `0.99` | Discount factor. |
| `--tau` | `1.0` | Target network update rate; `1.0` is a hard copy, as in the paper. |
| `--target-network-frequency` | `1000` | Steps between target network updates. |
| `--batch-size` | `32` | Transitions sampled from the buffer per update. |
| `--start-e` | `1` | Initial epsilon. |
| `--end-e` | `0.01` | Final epsilon. |
| `--exploration-fraction` | `0.10` | Fraction of `--total-timesteps` over which epsilon is annealed. |
| `--learning-starts` | `80000` | Step at which gradient updates begin. |
| `--train-frequency` | `4` | Steps between gradient updates. |

### Defaults compared with the paper

The values below are from the paper's hyperparameter table (Extended Data Table 1). The paper
counts some quantities in emulator frames and others in agent actions or parameter updates, so
note the units when comparing. With a frame skip of 4, one environment step here is 4 frames.

| Quantity | Paper | This repository |
| --- | --- | --- |
| Minibatch size | 32 | 32 |
| Replay memory size | 1,000,000 | 1,000,000 |
| Agent history length | 4 frames | 4 (the network expects 4 channels; stacking not yet applied) |
| Action repeat | 4 | not yet applied |
| Discount factor | 0.99 | 0.99 |
| Update frequency | every 4 actions | every 4 steps |
| Target network update | every 10,000 parameter updates | every 1,000 steps (250 updates) |
| Optimizer | RMSProp, lr 0.00025, momentum 0.95 | lr 1e-4; optimizer not yet written |
| Initial / final exploration | 1.0 / 0.1 | 1.0 / 0.01 |
| Final exploration frame | 1,000,000 | step 1,000,000 (`0.10 × 10M`) |
| Replay start size | 50,000 | 80,000 |
| No-op max | 30 | not yet applied |
| Training length | 50 million frames | 10 million steps (40 million frames at frame skip 4) |

The defaults follow the later CleanRL Atari DQN setup rather than the paper's table, which is why
the learning rate, final epsilon, target update interval, and replay start size differ.

---

## Layout

```text
dqn-atari/
├── algorithm.py        # arguments, environment setup, and (soon) the training loop
├── src/
│   └── agent.py        # QNetwork: the Nature DQN convolutional network
├── pyproject.toml      # Poetry dependency declaration, plus the `hub` extra
└── requirements.txt    # pip-installable freeze of the same versions
```

---

## References

- Mnih et al., *Human-level control through deep reinforcement learning* (2015).
  [Nature 518, 529–533](https://www.nature.com/articles/nature14236)
- Mnih et al., *Playing Atari with Deep Reinforcement Learning* (2013).
  [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
- van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning* (2015).
  [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)
- Machado et al., *Revisiting the Arcade Learning Environment* (2018).
  [arXiv:1709.06009](https://arxiv.org/abs/1709.06009)

## License

MIT, see [LICENSE](../LICENSE) at the repository root. This directory is original work;
provenance for the repository as a whole is recorded in [NOTICE](../NOTICE).
