"""Run several algorithms and seeds at once.

    python experiments.py --algos dqn ppo --seeds 1 2 3 --track
    python experiments.py --algos dqn ppo grpo --seeds 1 2 3 --track --save-model --hf-job --hf-flavor l4x1

Each (algorithm, seed) pair becomes one `python algorithm.py <algo> --seed <seed> ...` run. Any flag
not listed below is forwarded to every run, including `--hf-*` flags: with `--hf-job`, every run is
submitted as its own Hugging Face Job and they train in parallel there. Without it, runs train on
this machine as parallel processes, each logging to `logs/{run}.log`.

Flags that only one algorithm understands go in `--dqn-args`, `--ppo-args`, or `--grpo-args`.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from hf_jobs.launcher import parse_launcher_args
from src.algorithms import ALGORITHMS

PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algos", nargs="+", choices=list(ALGORITHMS), default=list(ALGORITHMS),
                        help="algorithms to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1],
                        help="seeds to run for every algorithm")
    parser.add_argument("--max-parallel", type=int, default=os.cpu_count() or 1,
                        help="maximum number of local runs training at the same time")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands without running them")
    for name in ALGORITHMS:
        parser.add_argument(f"--{name}-args", type=str, default="",
                            help=f"extra flags passed only to {name} runs, as one quoted string")
    args, forwarded = parser.parse_known_args()
    return args, forwarded


def build_commands(args, forwarded):
    commands = []
    for algo in args.algos:
        if not getattr(ALGORITHMS[algo], "IMPLEMENTED", True):
            print(f"Skipping {algo}: not implemented yet.")
            continue
        algo_args = shlex.split(getattr(args, f"{algo}_args"))
        for seed in args.seeds:
            name = f"{algo}__seed{seed}"
            command = [sys.executable, "algorithm.py", algo, "--seed", str(seed), *forwarded, *algo_args]
            commands.append((name, command))
    return commands


def run_locally(commands, max_parallel):
    """Run commands as parallel subprocesses, at most `max_parallel` at a time; return failed run names."""
    os.makedirs(PROJECT_DIR / "logs", exist_ok=True)
    pending = list(commands)
    running = {}
    failed = []
    while pending or running:
        while pending and len(running) < max_parallel:
            name, command = pending.pop(0)
            log_path = PROJECT_DIR / "logs" / f"{name}.log"
            log = open(log_path, "w", encoding="utf-8")
            running[name] = (subprocess.Popen(command, cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT), log)
            print(f"started  {name}  (log: {log_path.relative_to(PROJECT_DIR)})")
        for name, (process, log) in list(running.items()):
            if process.poll() is not None:
                log.close()
                del running[name]
                status = "done" if process.returncode == 0 else f"FAILED (exit {process.returncode})"
                print(f"finished {name}  {status}")
                if process.returncode != 0:
                    failed.append(name)
        time.sleep(1)
    return failed


def submit_jobs(commands):
    """Submit each run as its own Hugging Face Job; submission is quick, so this runs them in turn."""
    failed = []
    for name, command in commands:
        print(f"\n=== {name}")
        if subprocess.run(command, cwd=PROJECT_DIR).returncode != 0:
            failed.append(name)
    return failed


if __name__ == "__main__":
    args, forwarded = parse_args()
    commands = build_commands(args, forwarded)
    for name, command in commands:
        print(f"{name}: {shlex.join(command[1:])}")
    if args.dry_run or not commands:
        raise SystemExit(0)

    hf_job = parse_launcher_args(forwarded)[0].hf_job
    failed = submit_jobs(commands) if hf_job else run_locally(commands, args.max_parallel)
    print(f"\n{len(commands) - len(failed)}/{len(commands)} runs {'submitted' if hf_job else 'succeeded'}.")
    if failed:
        print("Failed: " + ", ".join(failed))
        raise SystemExit(1)
