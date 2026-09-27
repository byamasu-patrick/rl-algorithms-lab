"""Run a training script on Hugging Face Jobs instead of the local machine.

Call `launch()` at the top of a script's `__main__` block. When the command line
contains `--hf-job`, the script's project folder is uploaded to a private bucket,
a Job is submitted that installs the project's dependencies and runs the same
command there, and the local process exits. Without `--hf-job` it does nothing,
so the script runs locally exactly as before.

Every `--hf-*` flag below is removed from `sys.argv` before the script's own
argument parser runs, so the script never needs to know about them.
"""

import argparse
import netrc
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex

from hf_jobs.config import DEFAULT_EXCLUDE, JobConfig, resolve_config

MOUNT_PATH = "/bucket"
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


TRUE_VALUES = ("y", "yes", "t", "true", "on", "1")
FALSE_VALUES = ("n", "no", "f", "false", "off", "0")
# Only these may follow a switch as a separate argument (`--hf-job false`). Anything else, such as a
# Hydra override (`--hf-job +algorithm=ia2c`) or a number, is left for the script.
SEPARATE_BOOL_VALUES = ("true", "false", "yes", "no", "on", "off")

# flag -> (attribute name, takes a value); switches default to False, value options to None
LAUNCHER_FLAGS = {
    "--hf-job": ("hf_job", False),           # submit this run to Hugging Face Jobs
    "--hf-namespace": ("hf_namespace", True),  # user or org the Job runs and is billed under
    "--hf-flavor": ("hf_flavor", True),      # Job hardware, e.g. cpu-upgrade, t4-small, l4x1
    "--hf-timeout": ("hf_timeout", True),    # maximum Job duration, e.g. 90m, 12h
    "--hf-image": ("hf_image", True),        # Docker image the Job runs in
    "--hf-follow": ("hf_follow", False),     # stream the Job logs until it finishes
    "--hf-dry-run": ("hf_dry_run", False),   # print what would be uploaded and submitted, then exit
}


def _str_to_bool(flag, value):
    lowered = value.lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    raise SystemExit(f"{flag}: invalid truth value {value!r}")


def parse_launcher_args(argv):
    """Split `argv` into (launcher options, remaining script arguments).

    Only the exact flags in LAUNCHER_FLAGS are taken, as `--flag value` or `--flag=value`; every other
    argument is returned unchanged and in order, so argparse/tyro flags and Hydra overrides both pass
    through. Arguments after `--` are never read. Value options left unset are None, so the project's
    configuration can supply them.
    """
    argv = list(argv)
    opts = {dest: (None if takes_value else False) for dest, takes_value in LAUNCHER_FLAGS.values()}
    rest = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--":
            rest.extend(argv[i:])
            break
        flag, has_inline, inline = token.partition("=")
        if flag not in LAUNCHER_FLAGS:
            rest.append(token)
            i += 1
            continue

        dest, takes_value = LAUNCHER_FLAGS[flag]
        following = argv[i + 1] if i + 1 < len(argv) else None
        if takes_value:
            if has_inline:
                opts[dest] = inline
            elif following is None or following.startswith("-"):
                raise SystemExit(f"{flag} expects a value, e.g. {flag}=<value>")
            else:
                opts[dest] = following
                i += 1
        elif has_inline:
            opts[dest] = _str_to_bool(flag, inline)
        elif following is not None and following.lower() in SEPARATE_BOOL_VALUES:
            opts[dest] = _str_to_bool(flag, following)
            i += 1
        else:
            opts[dest] = True
        i += 1
    return argparse.Namespace(**opts), rest


def collect_files(folder, exclude=()):
    """Yield (path, posix path relative to `folder`) for every file to upload."""
    excluded = set(DEFAULT_EXCLUDE) | set(exclude)
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        if any(part in excluded or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_file() and path.suffix not in EXCLUDED_SUFFIXES:
            yield path, relative.as_posix()


def launcher_source_dir(project_dir):
    """This package's source checkout, when it sits next to `project_dir` (installed with `-e ../...`).

    Returned so it can be uploaded alongside the project and its relative path still resolves in the
    Job. A regular (PyPI) install returns None: the Job installs the package like any other dependency.
    """
    source = Path(__file__).resolve().parent.parent
    if (source / "pyproject.toml").exists() and source.parent == project_dir.parent and source != project_dir:
        return source
    return None


def _wandb_api_key():
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    for name in (".netrc", "_netrc"):
        path = Path.home() / name
        if path.exists():
            try:
                auth = netrc.netrc(str(path)).authenticators("api.wandb.ai")
            except netrc.NetrcParseError:
                continue
            if auth:
                return auth[2]
    return None


def collect_secrets(config, token):
    """HF token, W&B key when one is set up locally, and the variables listed in `config.secrets`."""
    secrets = {"HF_TOKEN": token}
    wandb_key = _wandb_api_key()
    if wandb_key:
        secrets["WANDB_API_KEY"] = wandb_key
    for name in config.secrets:
        if name not in os.environ:
            raise SystemExit(f"Secret {name!r} is listed in [tool.hf-jobs] secrets but is not set locally.")
        secrets[name] = os.environ[name]
    return secrets


def install_command(config, project_dir):
    if config.install:
        return config.install
    if (project_dir / "requirements.txt").exists():
        return "pip install --no-cache-dir -r requirements.txt"
    if (project_dir / "pyproject.toml").exists():
        return "pip install --no-cache-dir ."
    return "true"


def remote_command(config, project_dir, script_name, script_args):
    """Bash script run in the Job: copy code off the mount, install, run, copy outputs back."""
    run = shlex.join(["python", script_name, *script_args])
    outputs = " ".join(shlex.quote(d) for d in config.outputs)
    # Code is copied off the bucket mount so installs and training write to local disk; the output
    # directories are copied back afterwards, whether or not the script succeeded.
    return "\n".join([
        "set -u",
        f"cp -r {MOUNT_PATH}/. /workspace",
        f"cd /workspace/{shlex.quote(project_dir.name)}",
        f"{install_command(config, project_dir)} || exit 1",
        run,
        "status=$?",
        f"mkdir -p {MOUNT_PATH}/outputs",
        f"for d in {outputs}; do [ -d \"$d\" ] && cp -r \"$d\" {MOUNT_PATH}/outputs/; done",
        "exit $status",
    ])


def launch():
    """Submit the running script to Hugging Face Jobs when `--hf-job` is given; otherwise return."""
    opts, script_args = parse_launcher_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *script_args]
    if not opts.hf_job:
        return

    from huggingface_hub import HfApi, Volume, get_token

    script_path = Path(sys.argv[0]).resolve()
    project_dir = script_path.parent
    config = resolve_config(project_dir, {
        "namespace": opts.hf_namespace,
        "flavor": opts.hf_flavor,
        "timeout": opts.hf_timeout,
        "image": opts.hf_image,
    })

    token = get_token()
    if token is None:
        raise SystemExit("Not logged in to Hugging Face. Run `hf auth login` first.")
    api = HfApi(token=token)
    namespace = config.namespace or api.whoami()["name"]

    folders = {project_dir.name: project_dir}
    source = launcher_source_dir(project_dir)
    if source is not None:
        folders[source.name] = source

    run_id = f"{project_dir.name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{token_hex(3)}"
    bucket_id = f"{namespace}/{config.bucket}"
    # output folders are produced by the run, so a local copy of them is never uploaded
    excluded = [*config.exclude, *config.outputs]
    uploads = [
        (path, f"{run_id}/{name}/{relative}")
        for name, folder in folders.items()
        for path, relative in collect_files(folder, excluded)
    ]
    command = remote_command(config, project_dir, script_path.name, script_args)
    secrets = collect_secrets(config, token)

    print(f"namespace : {namespace}")
    print(f"flavor    : {config.flavor}  (timeout {config.timeout or 'default'}, image {config.image})")
    print(f"code      : {len(uploads)} files -> bucket {bucket_id}/{run_id}")
    print(f"secrets   : {', '.join(secrets)}")
    print(f"command   : {shlex.join(['python', script_path.name, *script_args])}")

    if opts.hf_dry_run:
        for _, remote in uploads:
            print(f"  {remote}")
        print("\n--- job script ---\n" + command)
        sys.exit(0)

    api.create_bucket(bucket_id=bucket_id, private=True, exist_ok=True)
    api.batch_bucket_files(bucket_id=bucket_id, add=uploads)

    job = api.run_job(
        image=config.image,
        command=["bash", "-c", command],
        env={"PYTHONUNBUFFERED": "1"},
        secrets=secrets,
        flavor=config.flavor,
        timeout=config.timeout,
        labels={"project": project_dir.name, "run": run_id},
        volumes=[Volume(type="bucket", source=bucket_id, mount_path=MOUNT_PATH, path=run_id, read_only=False)],
        namespace=namespace,
    )
    print(f"\nJob submitted: {job.url}")
    print(f"Outputs will be copied to bucket {bucket_id}/{run_id}/outputs")
    print(f"Logs: hf jobs logs {job.id} --namespace {namespace}")

    if opts.hf_follow:
        for line in api.fetch_job_logs(job_id=job.id, namespace=namespace, follow=True):
            print(line)
        print(f"Job finished: {api.inspect_job(job_id=job.id, namespace=namespace).status}")
    sys.exit(0)
