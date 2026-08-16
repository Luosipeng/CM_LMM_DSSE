"""Create reproducible figures from generated samples and experiment outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cm_llm.config import load_config
from cm_llm.training import prepare_experiment_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--history", default="outputs/tiny_reference_v2/history.json")
    parser.add_argument("--output-dir", default="docs/figures")
    args = parser.parse_args()
    config = load_config(args.config)
    data = prepare_experiment_data(config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    values = data["values"]
    time_hours = np.arange(min(192, len(values))) * config["data"]["sample_minutes"] / 60
    plt.figure(figsize=(9, 4.5))
    for index, bus in enumerate(data["sensor_buses"]):
        plt.plot(time_hours, values[: len(time_hours), index, 0], linewidth=1, label=f"Bus {bus}")
    plt.xlabel("Time (hour)")
    plt.ylabel("Voltage magnitude (p.u.)")
    plt.grid(alpha=0.25)
    plt.legend(ncol=5, fontsize=7)
    plt.tight_layout()
    plt.savefig(output / "sensor_voltage_timeseries.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 5))
    image = plt.imshow(data["graph"].adjacency, cmap="viridis", vmin=0, vmax=1)
    plt.xticks(range(len(data["sensor_buses"])), data["sensor_buses"])
    plt.yticks(range(len(data["sensor_buses"])), data["sensor_buses"])
    plt.xlabel("Target sensor bus")
    plt.ylabel("Source sensor bus")
    plt.colorbar(image, label="Causal weight")
    plt.tight_layout()
    plt.savefig(output / "causal_adjacency.png", dpi=180)
    plt.close()

    relation_strength = np.abs(data["graph"].lag_coefficients).sum(axis=(0, 1))
    relation_strength /= max(float(relation_strength.max()), 1e-8)
    plt.figure(figsize=(7, 6))
    image = plt.imshow(relation_strength, cmap="magma", vmin=0, vmax=1)
    labels = data["feature_names"]
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Target variable")
    plt.ylabel("Source variable")
    plt.colorbar(image, label="Normalized block coefficient strength")
    plt.tight_layout()
    plt.savefig(output / "variable_relation_strength.png", dpi=180)
    plt.close()

    history_path = Path(args.history)
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        epochs = [item["epoch"] for item in history]
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, [item["loss"] for item in history], label="Training")
        if "validation_loss" in history[0]:
            plt.plot(epochs, [item["validation_loss"] for item in history], label="Validation")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction objective")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "training_curve.png", dpi=180)
        plt.close()

    print(output.resolve())


if __name__ == "__main__":
    main()
