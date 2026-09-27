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
# Folder of this project inside the GitHub repository, e.g. "dqn-atari".
PROJECT_DIR = Path(__file__).resolve().parent.name


def model_card(args: argparse.Namespace, repo_id: str, algo_name: str, command: str) -> str:
    """Model card body; `command` is the training command line, e.g. `python algorithm.py --track`."""
    code_url = f"{GITHUB_REPO_URL}/tree/master/{PROJECT_DIR}"
    checkpoint_file = f"{args.exp_name}.cleanrl_model"
    return f"""
# **{algo_name}** Agent Playing **{args.env_id}**

This is a trained model of a {algo_name} agent playing {args.env_id}, trained by
[{AUTHOR_NAME}]({AUTHOR_HF_URL}) with a from-scratch implementation. The training code is in
[rl-algorithms-lab/{PROJECT_DIR}]({code_url}).

## Get Started

Clone the repository and install the project (Python 3.10 or 3.11):

```bash
git clone {GITHUB_REPO_URL}.git
cd rl-algorithms-lab/{PROJECT_DIR}
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then download and evaluate this checkpoint from the `{PROJECT_DIR}` directory:

```python
from huggingface_hub import hf_hub_download

from algorithm import make_env
from eval import evaluate
from src.agent import QNetwork

model_path = hf_hub_download(repo_id="{repo_id}", filename="{checkpoint_file}")
evaluate(model_path, make_env, "{args.env_id}", eval_episodes=10, run_name="eval",
         Model=QNetwork, device="cpu", capture_video=False)
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

The single-file layout and training loop follow [CleanRL](https://github.com/vwxyzjn/cleanrl)'s
`dqn_atari.py`.
"""


def card_metadata(args: argparse.Namespace, algo_name: str, mean_reward: str) -> dict:
    """Model card YAML header; `mean_reward` is formatted as "mean +/- std"."""
    from huggingface_hub.repocard import metadata_eval_result

    metadata = {
        "tags": [
            args.env_id,
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
    folder_path: str,
    video_folder_path: str = "",
    revision: str = "main",
    create_pr: bool = False,
    private: bool = False,
):
    # Step 1: lazy import and create / read a huggingface repo
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    from huggingface_hub.repocard import metadata_save

    api = HfApi()
    repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
        private=private,
    )
    # parse the default entity
    entity, repo = repo_url.split("/")[-2:]
    repo_id = f"{entity}/{repo}"

    # Step 2: clean up data
    # delete previous tfevents and mp4 files
    operations = [
        CommitOperationDelete(path_in_repo=file)
        for file in api.list_repo_files(repo_id=repo_id)
        if ".tfevents" in file or file.endswith(".mp4")
    ]

    # Step 3: Generate the model card
    algorithm_variant_filename = sys.argv[0].split("/")[-1]
    command = " ".join(["python", algorithm_variant_filename, *sys.argv[1:]])
    readme_path = Path(folder_path) / HUGGINGFACE_README_FILE_NAME
    readme = model_card(args, repo_id, algo_name, command)
    mean_reward = f"{np.average(episodic_returns):.2f} +/- {np.std(episodic_returns):.2f}"
    metadata = card_metadata(args, algo_name, mean_reward)

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    metadata_save(readme_path, metadata)

    # fetch mp4 files
    if video_folder_path:
        # Push all video files
        video_files = list(Path(video_folder_path).glob("*.mp4"))
        operations += [CommitOperationAdd(path_or_fileobj=str(file), path_in_repo=str(file)) for file in video_files]
        # Push latest one in root directory
        latest_file = max(video_files, key=lambda file: int("".join(filter(str.isdigit, file.stem))))
        operations.append(
            CommitOperationAdd(path_or_fileobj=str(latest_file), path_in_repo=HUGGINGFACE_VIDEO_PREVIEW_FILE_NAME)
        )

    # fetch folder files
    operations += [
        CommitOperationAdd(path_or_fileobj=str(item), path_in_repo=str(item.relative_to(folder_path)))
        for item in Path(folder_path).glob("*")
    ]

    # fetch source code
    operations.append(CommitOperationAdd(path_or_fileobj=sys.argv[0], path_in_repo=sys.argv[0].split("/")[-1]))

    # upload the project's dependency files; poetry.lock only exists after `poetry lock`
    project_root = Path(__file__).parent
    for dependency_file in ("pyproject.toml", "poetry.lock", "requirements.txt"):
        if (project_root / dependency_file).exists():
            operations.append(CommitOperationAdd(path_or_fileobj=str(project_root / dependency_file), path_in_repo=dependency_file))

    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message="pushing model",
        revision=revision,
        create_pr=create_pr,
    )
    print(f"Model pushed to {repo_url}")
    return repo_url