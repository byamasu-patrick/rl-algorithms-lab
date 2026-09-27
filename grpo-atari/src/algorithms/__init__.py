"""Algorithm registry. Each module provides:

- `DESCRIPTION`, `DISPLAY_NAME`: subcommand help and the name used on model cards;
- `add_args(parser)`: algorithm-specific flags;
- `validate_args(args)`: flag checks and runtime-computed sizes;
- `train(args, run, device)`: the training loop, ending with `run.save_evaluate_upload(...)`;
- optionally `IMPLEMENTED = False`, which makes `algorithm.py` refuse to start the run.
"""

from src.algorithms import dqn, grpo, ppo

ALGORITHMS = {
    "dqn": dqn,
    "ppo": ppo,
    "grpo": grpo,
}
