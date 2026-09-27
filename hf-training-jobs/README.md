# hf-jobs-launch

Run any Python training script on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)
by adding one flag to the command you already use.

```bash
python train.py --epochs 10                        # runs on this machine
python train.py --epochs 10 --hf-job               # same run, on Hugging Face
python train.py --epochs 10 --hf-job --hf-flavor a10g-small
```

There are no Dockerfiles or job specs to write, and your script does not change beyond one line.

## Install

From PyPI:

```bash
pip install hf-jobs-launch
```

Or from GitHub, for the latest code or a specific commit or tag:

```bash
pip install "hf-jobs-launch @ git+https://github.com/byamasu-patrick/rl-algorithms-lab.git#subdirectory=hf-training-jobs"
pip install "hf-jobs-launch @ git+https://github.com/byamasu-patrick/rl-algorithms-lab.git@<tag-or-commit>#subdirectory=hf-training-jobs"
```

Then log in once; the token is used to submit Jobs:

```bash
hf auth login
```

Requires Python 3.10+ and a Hugging Face account that can run Jobs (a PRO account, or an
organization with Jobs enabled).

## Use

Call `launch()` first thing in your script's `__main__` block, before your own argument parsing:

```python
from hf_jobs import launch

if __name__ == "__main__":
    launch()  # with --hf-job, submits this run to Hugging Face Jobs and exits
    args = parse_args()
    train(args)
```

Importing the module works too:

```python
from hf_jobs import launcher

if __name__ == "__main__":
    launcher.launch()
```

`launch()` removes every `--hf-*` flag from `sys.argv` before returning, so your parser never sees
them. Without `--hf-job` (or with `--hf-job false`) it returns immediately and the script runs
locally as usual.

### Hydra and other command-line styles

`launch()` doesn't depend on how your script reads its arguments. argparse, tyro and
[Hydra](https://hydra.cc/) scripts all work the same way:

- **Exact matches only.** Only the exact flags in [Flags](#flags) are read. Everything else stays in
  its original order, including look-alike flags such as `--hf-entity`, Hydra overrides such as
  `+algorithm=ia2c` and short options such as `-m`.
- **Switches.** `--hf-job`, `--hf-follow` and `--hf-dry-run` only consume the next argument when it
  is `true`, `false`, `yes`, `no`, `on` or `off`. `--hf-job +algorithm=ia2c` therefore submits the
  Job and leaves the override for Hydra. For any other value, use the `=` form: `--hf-job=0`.
- **`--`.** Arguments after `--` are passed through untouched.

For a Hydra app, call `launch()` before the `@hydra.main` function:

```python
@hydra.main(config_path="configs", config_name="default", version_base="1.3")
def main(cfg): ...

if __name__ == "__main__":
    launch()
    main()
```

A Hydra multirun (`-m`) runs all its combinations one after another inside one Job. Hydra's
`outputs/` and `multirun/` folders are copied back by default (see [`outputs`](#configure)).

If the script imports its own folder as a package (e.g. `from ac.model import ...` in a script
inside `ac/`), that works too. The folder keeps its name in the Job and sits at
`/workspace/<folder>`, so put `/workspace` on `sys.path` the same way you would locally.

Add the package to your project's dependencies. The Job installs them and runs the same script, so
it imports `hf_jobs` there too. Either source works in `requirements.txt`:

```text
hf-jobs-launch==0.2.0
# or
hf-jobs-launch @ git+https://github.com/byamasu-patrick/rl-algorithms-lab.git@<tag-or-commit>#subdirectory=hf-training-jobs
```

and likewise in `pyproject.toml`:

```toml
[project]
dependencies = ["hf-jobs-launch==0.2.0"]
```

Installing from GitHub inside the Job needs `git` in the image. The default `python:3.x` images
have it; the `-slim` variants do not.

## What `--hf-job` does

1. Uploads the folder containing your script to a private bucket, `{namespace}/hf-training-jobs`,
   under a new subfolder per run. Virtual environments, caches, `.git`, common output folders
   (`runs`, `videos`, `wandb`, `logs`, `outputs`, `multirun`) and every folder listed in `outputs`
   are skipped.
2. Submits a Job with that subfolder mounted at `/bucket`. The Job copies the code to local disk,
   installs dependencies, and runs `python <your script> <your arguments>` without the `--hf-*`
   flags.
3. When the script exits, successfully or not, copies the output folders back to `/bucket/outputs`,
   so logs and checkpoints outlive the Job. The defaults are `runs/` and `videos/` (CleanRL-style
   scripts) and `outputs/` and `multirun/` (Hydra); folders that don't exist are skipped.
4. Prints the Job URL and exits locally.

Dependencies are installed with `pip install -r requirements.txt` if the project has one, otherwise
`pip install .`. Set `install` (below) for anything else, e.g. `uv sync`.

### Secrets

These are passed to the Job as secrets, never as plain environment variables:

| Secret | Source |
| --- | --- |
| `HF_TOKEN` | your local Hugging Face login; lets the Job mount the bucket and push to the Hub as you |
| `WANDB_API_KEY` | `WANDB_API_KEY`, or the key `wandb login` saved in `~/.netrc` / `~/_netrc`, when present |
| anything in `secrets` | local environment variables listed in the configuration |

The Job acts with your token's permissions. Use a
[fine-grained token](https://huggingface.co/settings/tokens) limited to what the Job needs if that
matters to you.

## Configure

Per-project defaults go in the project's `pyproject.toml`:

```toml
[tool.hf-jobs]
namespace = "my-org"          # run and bill Jobs under an organization (default: your account)
flavor = "l4x1"               # hardware (default: cpu-basic); see `hf jobs hardware`
timeout = "24h"               # default: the Hugging Face default, 30 minutes
image = "python:3.11"         # default: python:3.12; needs bash, cp and pip
install = "pip install -r requirements.txt"   # default: chosen automatically, see above
outputs = ["runs", "checkpoints"]  # folders copied back and never uploaded
                                   # (default: runs, videos, outputs, multirun)
exclude = ["data"]            # extra folders not to upload
secrets = ["OPENAI_API_KEY"]  # local environment variables passed as secrets
bucket = "hf-training-jobs"   # bucket name, created in the namespace
```

Settings resolve in this order, highest first:

1. command-line flags (`--hf-namespace`, `--hf-flavor`, `--hf-timeout`, `--hf-image`),
2. environment variables `HF_JOBS_NAMESPACE`, `HF_JOBS_FLAVOR`, `HF_JOBS_TIMEOUT`, `HF_JOBS_IMAGE`,
   `HF_JOBS_INSTALL`, `HF_JOBS_BUCKET`,
3. `[tool.hf-jobs]` in the project's `pyproject.toml`,
4. the defaults above.

An unknown key in `[tool.hf-jobs]` is an error, so a typo such as `flavour` fails immediately instead
of being ignored.

## Flags

| Flag | Description |
| --- | --- |
| `--hf-job` | Submit to Hugging Face Jobs. `--hf-job false` runs locally. |
| `--hf-namespace` | User or organization the Job runs and is billed under. |
| `--hf-flavor` | Hardware, e.g. `cpu-upgrade`, `t4-small`, `a10g-small`, `l4x1`, `a100-large`. |
| `--hf-timeout` | Maximum duration, e.g. `90m`, `12h`. The Job is stopped when it is reached. |
| `--hf-image` | Docker image. |
| `--hf-follow` | Stream the Job logs in the terminal until it finishes. |
| `--hf-dry-run` | Print the files, secret names, and Job script, then exit without uploading anything. |

Value options take `--hf-flavor l4x1` or `--hf-flavor=l4x1`. The three switches accept an explicit
value as described in [Hydra and other command-line styles](#hydra-and-other-command-line-styles).

## Monitor

```bash
hf jobs ps --namespace my-org
hf jobs logs <job-id> --namespace my-org
hf jobs cancel <job-id> --namespace my-org
hf buckets ls my-org/hf-training-jobs
hf buckets sync hf://buckets/my-org/hf-training-jobs/<run-subfolder>/outputs ./outputs
```

## Developing against a local checkout

If the package is installed from a checkout that sits next to your project (for example
`-e ../hf-training-jobs` in `requirements.txt`), the checkout is uploaded alongside the project so
the same relative path installs in the Job. An install from PyPI or GitHub is installed in the Job
from the same source, like any other dependency.

```bash
pip install -e ".[test]"
pytest
```

## Changelog

### 0.2.0

- **Arguments:** the command line is parsed by exact flag name instead of argparse. Hydra overrides
  and other positional arguments right after `--hf-job` are no longer read as its value. Before
  this, `--hf-job +algorithm=ia2c` failed with `invalid truth value`. Switches only consume a
  separate `true`/`false`/`yes`/`no`/`on`/`off`, and arguments after `--` are never read. Existing
  argparse/tyro commands behave as before, and `parse_launcher_args` keeps its signature and return
  value.
- **Outputs:** `outputs` now defaults to `runs`, `videos`, `outputs` and `multirun`, so Hydra apps
  get their results back without configuration. Missing folders are still skipped, so CleanRL-style
  projects are unaffected.
- **Uploads:** folders listed in `outputs` are never uploaded, and `multirun` joins the default
  excludes.

### 0.1.0

- First release.

## License

MIT
