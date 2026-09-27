"""Train one algorithm: `python algorithm.py {dqn,ppo,grpo} [flags]`. See experiments.py to run many."""

from hf_jobs import launch
from src.algorithms import ALGORITHMS
from src.args import parse_args
from src.run import Run


if __name__ == "__main__":
    launch()  # with --hf-job, submits this run to Hugging Face Jobs and exits
    args = parse_args(ALGORITHMS)
    algorithm = ALGORITHMS[args.algo]
    if not getattr(algorithm, "IMPLEMENTED", True):
        raise SystemExit(f"{args.algo} is not implemented yet.")

    run = Run(args)
    if run.exists() and not args.overwrite:
        print(f"Run directory {run.name} already exists. Exiting.")
        raise SystemExit(0)

    device = run.start()
    try:
        algorithm.train(args, run, device)
    finally:
        run.close()
