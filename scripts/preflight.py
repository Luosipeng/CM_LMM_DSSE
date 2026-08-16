"""Read-only report plus one tiny forward/backward pass on generated data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cm_llm.config import load_config
from cm_llm.causal import graph_summary
from cm_llm.losses import ReconstructionLoss
from cm_llm.model.cmllm import CMLLM
from cm_llm.model.tiny import TinyCausalBackbone, TinyTokenizer
from cm_llm.prompts import build_prompts
from cm_llm.training import prepare_experiment_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    config = load_config(args.config)
    data = prepare_experiment_data(config)
    batch = next(iter(data["train_loader"]))
    model = CMLLM(
        TinyCausalBackbone(),
        TinyTokenizer(),
        len(data["feature_names"]),
        len(data["sensor_buses"]),
        adapter_hidden_size=32,
        dgp_layers=2,
        dropout=0.0,
    )
    graph_tensors = data["graph"].to_torch("cpu")
    prompts = build_prompts(
        batch["values"], batch["tasks"], data["sensor_buses"].tolist(), data["feature_names"]
    )
    prediction = model(batch["values"], graph_tensors, prompts, batch["valid_time"])
    training = config["training"]
    loss, parts = ReconstructionLoss(
        training["mask_loss_weight"], training["graph_temporal_weight"]
    )(
        prediction,
        batch["target"],
        batch["mask"],
        batch["valid_time"],
        graph_tensors,
    )
    loss.backward()
    report = {
        "samples": int(len(data["values"])),
        "sensors": data["sensor_buses"].tolist(),
        "features": data["feature_names"],
        "physical_edges": data["sensor_edges"].tolist(),
        "causal_graph": graph_summary(data["graph"]),
        "batch_shape": list(batch["values"].shape),
        "prediction_shape": list(prediction.shape),
        "loss": float(loss.detach()),
        "loss_parts": {key: float(value.detach()) for key, value in parts.items()},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
