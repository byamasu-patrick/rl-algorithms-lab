"""Per-run setup shared by every algorithm: naming, tracking, seeding, device, and checkpoints."""

import os
import random
import shutil

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.checkpoints import load_checkpoint, save_checkpoint


class Run:
    """Owns everything a training loop writes to: the TensorBoard writer, W&B, and `runs/{name}/`."""

    def __init__(self, args):
        self.args = args
        if args.run_name:
            self.name = args.run_name
        else:
            self.name = f"{args.exp_name}__{args.env_id.replace('/', '_').replace('-', '_').lower()}__{args.seed}"
        self.dir = f"runs/{self.name}"
        self.next_checkpoint = 0

    def exists(self):
        return os.path.exists(f"{self.dir}/config.yaml")

    def start(self):
        """Create the run directory and trackers, seed everything, and return the torch device."""
        args = self.args
        if self.exists():
            print(f"Run directory {self.name} already exists. Overwriting.")
            shutil.rmtree(self.dir)

        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=self.name,
                group=args.exp_name,
                tags=[args.algo, args.env_id],
                save_code=True,
            )

        self.writer = SummaryWriter(self.dir)
        self.writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))
        with open(f"{self.dir}/config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(vars(args), f)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        print(f"Run {self.name} on {device}")
        return device

    def load(self, device, **objects):
        """Initialize `objects` (modules/optimizers) from --checkpoint-load-path, if one was given."""
        if not self.args.checkpoint_load_path:
            return
        global_step, iteration = load_checkpoint(
            self.args.checkpoint_load_path, self.args.checkpoint_param_filters, device, **objects)
        print(f"Loaded checkpoint saved at step {global_step}, iteration {iteration}")
        self.writer.add_text("resume", f"|param|value|\n|-|-|\n|global_step|{global_step}|\n|iteration|{iteration}|")

    def maybe_checkpoint(self, global_step, iteration, **objects):
        """Save a checkpoint every --checkpoint-every env steps (starting at step 0)."""
        if self.args.checkpoint_every > 0 and global_step >= self.next_checkpoint:
            self.next_checkpoint = global_step + self.args.checkpoint_every
            save_checkpoint(self.dir, global_step, iteration, **objects)

    def model_path(self):
        return f"{self.dir}/{self.args.exp_name}.pt"

    def save_evaluate_upload(self, model, act, algo_name):
        """With --save-model: save `model`, evaluate it with `act(obs) -> actions`, and optionally upload it."""
        args = self.args
        if not args.save_model:
            return
        from src.evaluation import evaluate

        model_path = self.model_path()
        torch.save(model.state_dict(), model_path)
        print(f"model saved to {model_path}")

        eval_run_name = f"{self.name}-eval"
        episodic_returns = evaluate(args, eval_run_name, act, args.eval_episodes)
        for idx, episodic_return in enumerate(episodic_returns):
            self.writer.add_scalar("eval/episodic_return", episodic_return, idx)
        self.writer.add_scalar("eval/mean_episodic_return", float(np.mean(episodic_returns)), 0)

        if args.upload_model:
            from src.hub import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            self.writer.flush()
            push_to_hub(args, episodic_returns, repo_id, algo_name, type(model).__name__, model_path,
                        self.dir, f"videos/{eval_run_name}")

    def close(self):
        self.writer.close()
        if self.args.track:
            import wandb

            wandb.finish()
