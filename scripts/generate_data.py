"""Generate the configured case33bw time series through local MATPOWER."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cm_llm.config import load_config
from cm_llm.data.matpower import run_matpower_generation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    output = run_matpower_generation(
        data["output_path"],
        data["matpower_root"],
        args.steps or data["n_steps"],
        config["seed"],
        data["sample_minutes"],
    )
    print(output)


if __name__ == "__main__":
    main()

