"""Run the self-contained historical ACCESS-CM2 example."""

import argparse
import json
from pathlib import Path

from moses.config import Config, ROOT
from moses.workflow import check_inputs, run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true", help="Check input variables and historical track metadata, then exit.")
    parser.add_argument("--workers", type=int, default=1, help="Processes for synthetic track mapping (default: 1).")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs", help="Output directory (default: outputs/ beside this script).")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.check_inputs:
        print(json.dumps(check_inputs(), indent=2))
    else:
        run_demo(Config(), args.output.resolve(), args.workers)
        from plot_results import plot_results
        plot_results(args.output.resolve())


if __name__ == "__main__":
    main()
