# Reinforcement Learning Algorithms Lab

A modular research repository for implementing, reproducing, and extending reinforcement learning algorithms.

This repository is inspired by the paper **"Learning Without Critics? Revisiting GRPO in Classical Reinforcement Learning Environments"** and extends its ideas to investigate Group Relative Policy Optimization (GRPO) in broader reinforcement learning settings.

The current research focuses on:

- Reproducing the results of the original GRPO paper.
- Studying critic-free reinforcement learning algorithms.
- Extending GRPO to partially observable environments (POMDPs).
- Investigating different trajectory and group sampling strategies.
- Comparing return estimators, baseline methods, and variance reduction techniques.
- Developing modular implementations that facilitate algorithmic experimentation, reproducibility, and future research.

The repository emphasizes clean, modular implementations that are easy to understand, modify, and extend for reinforcement learning research.
## Without Docker

1. Install UV.
2. Install dependencies with `uv sync` and then `uv sync --extra mujoco`
3. Test with CartPole: `uv run python algorithm.py --no-track`
4. Run all missing experiments using all CPU cores: `bash launch_all_cpus.sh`
6. Alternatively, manually run the experiments with `bash experiment.sh <total_instances> <current_instance_index>`, e.g.:

```bash
bash experiment.sh 1 0  # to run all experiments sequentially
```

or 

```bash
# each in a different terminal instance (e.g. tmux):
bash experiment.sh 4 0
bash experiment.sh 4 1  
bash experiment.sh 4 2
bash experiment.sh 4 3
```

The `experiment.sh` script will first enumerate all experiments and then split them into the total number of instances and run the commands that are multiples of the current instance index.

## With Docker

1. `docker build -t grpo .`
2. `docker run -e WANDB_API_KEY=<key> grpo`

## Citing

If you use this code in your research, please cite:

```bibtex
@inproceedings{oliveira2025learning,
title={Learning Without Critics? Revisiting {GRPO} in Classical Reinforcement Learning Environments},
author={Bryan L. M. de Oliveira and Felipe V. Frujeri and Marcos P. C. M. Queiroz and Luana G. B. Martins and Telma W. de L. Soares and Luckeciano C. Melo},
booktitle={Latinx in AI @ NeurIPS 2025},
year={2025},
}
```