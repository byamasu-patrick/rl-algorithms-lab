"""Run any training script on Hugging Face Jobs by adding `--hf-job` to its command line."""

from hf_jobs.config import JobConfig, resolve_config
from hf_jobs.launcher import launch, parse_launcher_args

__version__ = "0.1.0"
__all__ = ["launch", "parse_launcher_args", "JobConfig", "resolve_config", "__version__"]
