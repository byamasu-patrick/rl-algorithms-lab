"""Hugging Face Hub upload with a model card crediting the author and linking the training code."""

import argparse
import sys
from pathlib import Path
from pprint import pformat
from typing import List

import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed

HUGGINGFACE_VIDEO_PREVIEW_FILE_NAME = "replay.mp4"
HUGGINGFACE_README_FILE_NAME = "README.md"

AUTHOR_NAME = "Byamasu Patrick Paul"
AUTHOR_HF_URL = "https://huggingface.co/byamasupatrick"
GITHUB_REPO_URL = "https://github.com/byamasu-patrick/rl-algorithms-lab"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Folder of this project inside the GitHub repository, e.g. "grpo-atari".
PROJECT_DIR = PROJECT_ROOT.name


def model_card(args: argparse.Namespace, repo_id: str, algo_name: str, network: str, checkpoint_file: str,
               command: str) -> str:
    """Model card body; `command` is the training command line, e.g. `python algorithm.py dqn --track`."""
    code_url = f"{GITHUB_REPO_URL}/tree/master/{PROJECT_DIR}"
    return f"""
# **{algo_name}** Agent Playing **{args.env_id}**

This is a trained model of a {algo_name} agent playing {args.env_id}, trained by
[{AUTHOR_NAME}]({AUTHOR_HF_URL}) with a from-scratch implementation. The training code is in
[rl-algorithms-lab/{PROJECT_DIR}]({code_url}), which trains DQN, PPO and GRPO on Atari with
identical preprocessing so the three can be compared.

## Get Started

Clone the repository and install the project (Python 3.10 or 3.11):

```bash
git clone {GITHUB_REPO_URL}.git
cd rl-algorithms-lab/{PROJECT_DIR}
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The checkpoint is the state dict of `{network}` in `src/algorithms/{args.algo}.py`. Build the
environments with `src.environment.make_vector_env` and the hyperparameters below, then load it:

```python
import torch
from huggingface_hub import hf_hub_download

state_dict = torch.load(hf_hub_download(repo_id="{repo_id}", filename="{checkpoint_file}"), map_location="cpu")
```

## Command to reproduce the training

From the `{PROJECT_DIR}` directory:

```bash
{command}
```

See the [project README]({code_url}) for the implementation details and all command-line flags.

## Hyperparameters

```python
{pformat(vars(args))}
```

## Acknowledgements

The single-file training loops follow [CleanRL](https://github.com/vwxyzjn/cleanrl)'s `dqn_atari.py`
and `ppo_atari.py`; the return and baseline variants follow *Learning Without Critics? Revisiting
GRPO in Classical Reinforcement Learning Environments* (de Oliveira et al., 2025).
"""


def card_metadata(args: argparse.Namespace, algo_name: str, mean_reward: str) -> dict:
    """Model card YAML header; `mean_reward` is formatted as "mean +/- std"."""
    from huggingface_hub.repocard import metadata_eval_result

    metadata = {
        "tags": [
            args.env_id,
            algo_name,
            "deep-reinforcement-learning",
            "reinforcement-learning",
            "custom-implementation",
        ],
    }
    eval = metadata_eval_result(
        model_pretty_name=algo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=mean_reward,
        dataset_pretty_name=args.env_id,
        dataset_id=args.env_id,
    )
    return {**metadata, **eval}


@retry(stop=stop_after_attempt(10), wait=wait_fixed(3))
def push_to_hub(
    args: argparse.Namespace,
    episodic_returns: List,
    repo_id: str,
    algo_name: str,
    network: str,
    model_path: str,
    folder_path: str,
    video_folder_path: str = "",
    revision: str = "main",
    create_pr: bool = False,
    private: bool = False,
):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    from huggingface_hub.repocard import metadata_save

    api = HfApi()
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
    entity, repo = repo_url.split("/")[-2:]
    repo_id = f"{entity}/{repo}"

    # Delete previous tfevents and mp4 files
    operations = [
        CommitOperationDelete(path_in_repo=file)
        for file in api.list_repo_files(repo_id=repo_id)
        if ".tfevents" in file or file.endswith(".mp4")
    ]

    # Model card
    command = " ".join(["python", Path(sys.argv[0]).name, *sys.argv[1:]])
    readme_path = Path(folder_path) / HUGGINGFACE_README_FILE_NAME
    mean_reward = f"{np.average(episodic_returns):.2f} +/- {np.std(episodic_returns):.2f}"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card(args, repo_id, algo_name, network, Path(model_path).name, command))
    metadata_save(readme_path, card_metadata(args, algo_name, mean_reward))

    # Videos, with the latest one as the preview
    video_files = list(Path(video_folder_path).glob("*.mp4")) if video_folder_path else []
    if video_files:
        operations += [CommitOperationAdd(path_or_fileobj=str(file), path_in_repo=file.as_posix()) for file in video_files]
        latest_file = max(video_files, key=lambda file: int("".join(filter(str.isdigit, file.stem)) or 0))
        operations.append(CommitOperationAdd(path_or_fileobj=str(latest_file), path_in_repo=HUGGINGFACE_VIDEO_PREVIEW_FILE_NAME))

    # Run directory files (model, TensorBoard logs, config, card); periodic checkpoints stay local
    operations += [
        CommitOperationAdd(path_or_fileobj=str(item), path_in_repo=item.name)
        for item in Path(folder_path).glob("*")
        if item.is_file()
    ]

    # Source code and dependency files
    for source in ["algorithm.py", "requirements.txt", "pyproject.toml", *(p.relative_to(PROJECT_ROOT).as_posix() for p in (PROJECT_ROOT / "src").rglob("*.py"))]:
        if (PROJECT_ROOT / source).exists():
            operations.append(CommitOperationAdd(path_or_fileobj=str(PROJECT_ROOT / source), path_in_repo=source))

    api.create_commit(repo_id=repo_id, operations=operations, commit_message="pushing model",
                      revision=revision, create_pr=create_pr)
    print(f"Model pushed to {repo_url}")
    return repo_url
