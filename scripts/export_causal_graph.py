"""Export selected block-graph relations to auditable CSV and JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cm_llm.causal import graph_summary
from cm_llm.config import load_config
from cm_llm.training import prepare_experiment_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output-dir", default="outputs/block_graph")
    parser.add_argument("--minimum-coefficient", type=float, default=1e-8)
    args = parser.parse_args()
    config = load_config(args.config)
    data = prepare_experiment_data(config)
    graph = data["graph"]
    buses = data["sensor_buses"]
    features = data["feature_names"]
    rows: list[dict[str, object]] = []
    for edge, (source, target) in enumerate(graph.edge_index.T):
        for lag in range(graph.lag_order):
            block = graph.lag_coefficients[edge, lag]
            for source_feature in range(graph.feature_count):
                for target_feature in range(graph.feature_count):
                    coefficient = float(block[source_feature, target_feature])
                    if abs(coefficient) <= args.minimum_coefficient:
                        continue
                    rows.append(
                        {
                            "source_bus": int(buses[source]),
                            "target_bus": int(buses[target]),
                            "lag": lag + 1,
                            "source_feature": features[source_feature],
                            "target_feature": features[target_feature],
                            "coefficient": coefficient,
                            "absolute_coefficient": abs(coefficient),
                            "edge_type": graph.edge_types[edge],
                        }
                    )
    rows.sort(key=lambda row: float(row["absolute_coefficient"]), reverse=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "variable_relations.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    summary_path = output / "graph_summary.json"
    summary_path.write_text(
        json.dumps(graph_summary(graph), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(csv_path.resolve())
    print(summary_path.resolve())


if __name__ == "__main__":
    main()
