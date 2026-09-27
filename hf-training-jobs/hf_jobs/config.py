"""Job settings, resolved from (highest first): `--hf-*` flags, `HF_JOBS_*` environment variables,
the project's `[tool.hf-jobs]` table in `pyproject.toml`, and the defaults below."""

import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# Directories never uploaded with the project code; `exclude` in the config adds to these.
DEFAULT_EXCLUDE = (
    ".venv", "venv", "env", "__pycache__", ".git", ".hg", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".ipynb_checkpoints", "node_modules", "wandb", "runs", "videos", "logs", "outputs", "multirun",
)


@dataclass
class JobConfig:
    namespace: str | None = None
    """User or organization the Job runs and is billed under; None means the token's own account."""
    flavor: str = "cpu-basic"
    """Job hardware, e.g. cpu-upgrade, t4-small, a10g-small, l4x1, a100-large."""
    timeout: str | None = None
    """Maximum Job duration, e.g. "90m", "24h"; None uses the Hugging Face default (30 minutes)."""
    image: str = "python:3.12"
    """Docker image the Job runs in; it needs `bash`, `cp`, and `pip`."""
    install: str | None = None
    """Shell command that installs dependencies; None picks `pip install -r requirements.txt`, then `pip install .`."""
    outputs: list[str] = field(default_factory=lambda: ["runs", "videos", "outputs", "multirun"])
    """Directories copied back to the bucket after the script exits, whether or not it succeeded, and
    never uploaded. Missing ones are skipped. The defaults cover CleanRL-style scripts (runs, videos) and
    Hydra apps (outputs, multirun)."""
    exclude: list[str] = field(default_factory=list)
    """Extra directory names to leave out of the upload."""
    secrets: list[str] = field(default_factory=list)
    """Names of local environment variables to pass to the Job as secrets."""
    bucket: str = "hf-training-jobs"
    """Private bucket, created in the Job namespace, holding uploaded code and outputs."""


# Settings that can be overridden by an environment variable, e.g. HF_JOBS_NAMESPACE=my-org.
ENV_SETTINGS = ("namespace", "flavor", "timeout", "image", "install", "bucket")


def read_pyproject(project_dir: Path) -> dict:
    """Return the `[tool.hf-jobs]` table of `project_dir/pyproject.toml`, or {}."""
    path = project_dir / "pyproject.toml"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        table = tomllib.load(f).get("tool", {}).get("hf-jobs", {})
    known = {f.name for f in fields(JobConfig)}
    unknown = set(table) - known
    if unknown:
        raise ValueError(f"Unknown keys in [tool.hf-jobs] of {path}: {', '.join(sorted(unknown))}. "
                         f"Valid keys: {', '.join(sorted(known))}.")
    return table


def resolve_config(project_dir: Path, overrides: dict | None = None, environ=None) -> JobConfig:
    """Merge defaults, pyproject, environment and `overrides` (non-None values win) into a JobConfig."""
    environ = os.environ if environ is None else environ
    values = read_pyproject(project_dir)
    for name in ENV_SETTINGS:
        env_value = environ.get(f"HF_JOBS_{name.upper()}")
        if env_value:
            values[name] = env_value
    for name, value in (overrides or {}).items():
        if value is not None:
            values[name] = value
    return JobConfig(**values)
