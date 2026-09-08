"""Run one configuration and print the row. For debugging a single point, not for sweeps."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import ExperimentRunner, RunConfig
from src.storage import ResultWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=[RunConfig.S1, RunConfig.S2])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="logistic_regression")
    parser.add_argument("--attack", required=True, choices=[RunConfig.POISONING, RunConfig.EVASION, RunConfig.BYZANTINE])
    parser.add_argument("--attack-mode", required=True)
    parser.add_argument("--intensity", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-clients", type=int, default=10)
    parser.add_argument("--partition", default="iid")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--output", default=None, help="append the row to this CSV; default prints only")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    common = dict(
        scenario=args.scenario,
        dataset=args.dataset,
        model=args.model,
        attack=args.attack,
        attack_mode=args.attack_mode,
        intensity=args.intensity,
        seed=args.seed,
    )
    if args.scenario == RunConfig.S1:
        return RunConfig(**common)
    return RunConfig(
        **common,
        n_clients=args.n_clients,
        partition=args.partition,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        dirichlet_alpha=args.dirichlet_alpha,
    )


def main() -> None:
    args = parse_args()
    row = ExperimentRunner().run(build_config(args))
    print(json.dumps(row, indent=2))
    if args.output:
        ResultWriter(args.output).append(row)


if __name__ == "__main__":
    main()
