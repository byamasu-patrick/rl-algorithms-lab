"""Checkpoint utilities for saving and loading model checkpoints (from revisiting-grpo)."""

import os
import re

import torch


def save_checkpoint(run_dir, global_step, iteration, **objects):
    """Save the state dicts of `objects` to `{run_dir}/checkpoints/step_{global_step}.pth`."""
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
    checkpoint_path = f"{run_dir}/checkpoints/step_{global_step}.pth"
    save_dict = {"global_step": global_step, "iteration": iteration}
    for key, obj in objects.items():
        save_dict[key] = obj.state_dict()
    torch.save(save_dict, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(checkpoint_path, param_filters=None, device=None, **objects):
    """Load state dicts into `objects`; with `param_filters[key]`, only matching parameters are loaded.

    Returns the checkpoint's (global_step, iteration), or (0, 0) when any parameters were filtered.
    """
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    filtered_any = False

    if not param_filters:
        param_filters = {}
    for key, module in objects.items():
        if key not in checkpoint:
            raise KeyError(
                f"Checkpoint does not contain required key '{key}'. "
                f"Available keys: {list(checkpoint.keys())}. "
                f"This may indicate checkpoint corruption or version mismatch."
            )
        module_state_dict = checkpoint[key]
        if param_filters.get(key):
            filtered_any = True
            regex = re.compile(param_filters[key])
            # Parameters not matching the regex keep the module's current values.
            filtered_state_dict = {k: v for k, v in module.state_dict().items() if not regex.match(k)}
            print("Not loading params: ", list(filtered_state_dict.keys()))
            module_state_dict.update(filtered_state_dict)
        module.load_state_dict(module_state_dict)

    if filtered_any:
        return 0, 0
    return checkpoint["global_step"], checkpoint["iteration"]
