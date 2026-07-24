"""Wandb utilities for initializing and managing Weights & Biases tracking."""
from traitlets import Any

def init_wandb(args: Any, run_name: str) -> None:
    """Initialize Weights & Biases if tracking is enabled."""

    if not args.track:
        return None
    
    import wandb

    api = wandb.Api()
    runs = api.runs(f"{args.wandb_entity}/{args.wandb_project_name}", filters={"display_name": run_name})

    if runs:
        print(f"Wandb run {run_name} already exists, skipping")
        exit(0)

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        group=args.exp_name,
        monitor_gym=True,
        save_code=True,
    )