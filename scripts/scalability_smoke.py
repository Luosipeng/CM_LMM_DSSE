"""Synthetic large-feeder smoke test for block-graph scaling behavior."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cm_llm.causal import discover_causal_graph, graph_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensors", type=int, default=100)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--features", type=int, default=6)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    values = np.zeros((args.steps, args.sensors, args.features), dtype=np.float64)
    values[0] = rng.normal(size=(args.sensors, args.features))
    for step in range(1, args.steps):
        local = 0.78 * values[step - 1]
        upstream = np.zeros_like(local)
        upstream[1:] = 0.12 * values[step - 1, :-1]
        values[step] = local + upstream + 0.15 * rng.normal(size=local.shape)
    explicit = np.column_stack(
        (np.arange(args.sensors - 1), np.arange(1, args.sensors))
    )
    positions = np.arange(args.sensors)
    electrical_distance = np.abs(positions[:, None] - positions[None, :]).astype(float)
    feature_names = [
        "vm_pu",
        "va_degree",
        "p_inj_mw",
        "q_inj_mvar",
        "p_upstream_mw",
        "q_upstream_mvar",
    ][: args.features]
    feature_names.extend(
        f"feature_{index}" for index in range(len(feature_names), args.features)
    )
    start = time.perf_counter()
    graph = discover_causal_graph(
        values,
        explicit,
        feature_names=feature_names,
        electrical_distance=electrical_distance,
        lag_order=2,
        max_neighbors=args.neighbors,
        group_lasso=0.1,
        l1_penalty=0.01,
        edge_threshold=0.02,
        max_iterations=80,
        n_jobs=args.n_jobs,
    )
    elapsed = time.perf_counter() - start
    dense_coefficients = args.sensors**2 * 2 * args.features**2
    report = graph_summary(graph)
    report.update(
        {
            "elapsed_seconds": elapsed,
            "dense_coefficients": dense_coefficients,
            "storage_reduction": dense_coefficients
            / max(report["stored_block_coefficients"], 1),
        }
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
