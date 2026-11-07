import argparse
import sys
from pathlib import Path
from pprint import pformat
from typing import List

import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed

HUGGINGFACE_VIDEO_PREVIEW_FILE_NAME = "replay.mp4"
HUGGINGFACE_README_FILE_NAME = "README.md"
PROJECT_DIR = Path(__file__).resolve().parent


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
    from huggingface_hub.repocard import metadata_eval_result, metadata_save

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
    algorithm_variant_filename = Path(sys.argv[0]).name
    model_card = f"""
# **{algo_name}** Agent Playing **{args.env_id}**

This is a trained model of a {algo_name} agent playing {args.env_id}, trained with a
[CleanRL](https://github.com/vwxyzjn/cleanrl)-style single-file implementation.

## Files

- `{args.exp_name}.cleanrl_model`: a `(actor_state_dict, critic_state_dict)` tuple saved with `torch.save`
- `{algorithm_variant_filename}`, `requirements.txt`, `pyproject.toml`: the training code and its pinned dependencies
- `events.out.tfevents.*`: TensorBoard logs; `replay.mp4`: the latest evaluation video

## Command to reproduce the training

```bash
curl -OL https://huggingface.co/{repo_id}/raw/main/{algorithm_variant_filename}
curl -OL https://huggingface.co/{repo_id}/raw/main/requirements.txt
pip install -r requirements.txt
python {algorithm_variant_filename} {" ".join(sys.argv[1:])}
```

# Hyperparameters
```python
{pformat(vars(args))}
```
    """
    readme_path = Path(folder_path) / HUGGINGFACE_README_FILE_NAME
    readme = model_card

    # metadata
    metadata = {}
    metadata["tags"] = [
        args.env_id,
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "custom-implementation",
    ]
    metadata["library_name"] = "cleanrl"
    eval = metadata_eval_result(
        model_pretty_name=algo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{np.average(episodic_returns):.2f} +/- {np.std(episodic_returns):.2f}",
        dataset_pretty_name=args.env_id,
        dataset_id=args.env_id,
    )
    metadata = {**metadata, **eval}

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    metadata_save(readme_path, metadata)

    # fetch mp4 files
    video_files = list(Path(video_folder_path).glob("*.mp4")) if video_folder_path else []
    if video_files:
        # Push all video files
        operations += [CommitOperationAdd(path_or_fileobj=str(file), path_in_repo=file.as_posix()) for file in video_files]
        # Push latest one in root directory
        latest_file = max(video_files, key=lambda file: int("".join(filter(str.isdigit, file.stem)) or 0))
        operations.append(
            CommitOperationAdd(path_or_fileobj=str(latest_file), path_in_repo=HUGGINGFACE_VIDEO_PREVIEW_FILE_NAME)
        )

    # fetch folder files
    operations += [
        CommitOperationAdd(path_or_fileobj=str(item), path_in_repo=item.relative_to(folder_path).as_posix())
        for item in Path(folder_path).glob("*")
        if item.is_file()
    ]

    # fetch source code
    operations.append(CommitOperationAdd(path_or_fileobj=sys.argv[0], path_in_repo=algorithm_variant_filename))

    # upload the dependency files that sit next to this script
    for name in ("requirements.txt", "pyproject.toml"):
        if (PROJECT_DIR / name).exists():
            operations.append(CommitOperationAdd(path_or_fileobj=str(PROJECT_DIR / name), path_in_repo=name))

    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message="pushing model",
        revision=revision,
        create_pr=create_pr,
    )
    print(f"Model pushed to {repo_url}")
    return repo_url
