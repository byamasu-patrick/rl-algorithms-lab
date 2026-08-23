
"""Checkpoint utilities for saving and loading model checkpoints."""

import torch
import re

checkpoint_counter = 0

def save_checkpoint(run_name, global_step, iteration, **objects):
    """Save a checkpoint of the training state."""
    global checkpoint_counter
    checkpoint_path = f"runs/{run_name}/checkpoint_{checkpoint_counter}.pth"
    save_dict = {"global_step": global_step, "iteration": iteration}
    for key, obj in objects.items():
        save_dict[key] = obj.state_dict()
    torch.save(save_dict, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    checkpoint_counter += 1


def load_checkpoint(checkpoint_path, param_filters=None, device=None, **objects):
    """Load a checkpoint and restore the training state."""

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
            print("Checkpoint params: ", checkpoint[key].keys())
            print("Unloading params not matching: ", param_filters[key])
            regex = re.compile(param_filters[key])
            filtered_state_dict = {k: v for k, v in module.state_dict().items() if not regex.match(k)}
            print("Unloaded params: ", filtered_state_dict.keys())
            module_state_dict.update(filtered_state_dict)
        module.load_state_dict(module_state_dict)

    if filtered_any:
        return 0, 0
    else:
        return checkpoint['global_step'], checkpoint['iteration']

