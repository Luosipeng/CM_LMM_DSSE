"""Evaluate a trained adapter and classical baselines on chronological test data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cm_llm.baselines import classical_reconstruction
from cm_llm.config import load_config
from cm_llm.metrics import radial_voltage_violation_rate, reconstruction_metrics
from cm_llm.model.cmllm import CMLLM, load_qwen_backbone
from cm_llm.model.tiny import TinyCausalBackbone, TinyTokenizer
from cm_llm.prompts import build_prompts
from cm_llm.training import prepare_experiment_data


def _model(config: dict, data: dict, kind: str, allow_download: bool) -> CMLLM:
    model_config = config["model"]
    if kind == "qwen":
        source = model_config["model_path"] or model_config["model_id"]
        processor, backbone = load_qwen_backbone(
            source,
            dtype=model_config["dtype"],
            use_lora=model_config["use_lora"],
            lora_rank=model_config["lora_rank"],
            lora_alpha=model_config["lora_alpha"],
            lora_dropout=model_config["lora_dropout"],
            local_files_only=not allow_download,
        )
        tokenizer = processor.tokenizer
    else:
        backbone, tokenizer = TinyCausalBackbone(), TinyTokenizer()
    return CMLLM(
        backbone,
        tokenizer,
        len(data["feature_names"]),
        len(data["sensor_buses"]),
        model_config["hidden_size"],
        model_config["dgp_layers"],
        model_config["dropout"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", choices=("qwen", "tiny"), default="qwen")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    data = prepare_experiment_data(config)
    model = _model(config, data, args.backbone, args.allow_download)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["trainable_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    # Missing keys are expected because frozen Qwen parameters are intentionally
    # absent from adapter-only checkpoints.
    del missing
    if args.backbone == "qwen":
        device = next(model.backbone.parameters()).device
        for module in (model.temporal, model.causal, model.to_llm, model.output_projection):
            module.to(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    model.eval()
    graph_tensors = data["graph"].to_torch(device)
    normalizer = data["standardizer"]
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    with torch.inference_mode():
        for batch_index, batch in enumerate(data["test_loader"]):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            values = batch["values"].to(device)
            prompts = build_prompts(
                values, batch["tasks"], data["sensor_buses"].tolist(), data["feature_names"]
            )
            prediction = model(
                values, graph_tensors, prompts, batch["valid_time"].to(device)
            )
            prediction_np = prediction.float().cpu().numpy()
            target_np = batch["target"].numpy()
            mask_np = batch["mask"].numpy()
            values_np = batch["values"].numpy()
            for row, task in enumerate(batch["tasks"]):
                predicted_real = normalizer.inverse_transform(prediction_np[row])
                target_real = normalizer.inverse_transform(target_np[row])
                model_metric = reconstruction_metrics(predicted_real, target_real, mask_np[row])
                baseline = classical_reconstruction(values_np[row], task)
                baseline_real = normalizer.inverse_transform(baseline)
                baseline_metric = reconstruction_metrics(baseline_real, target_real, mask_np[row])
                for key, value in model_metric.items():
                    scores[task][f"cmllm_{key}"].append(value)
                for key, value in baseline_metric.items():
                    scores[task][f"baseline_{key}"].append(value)
                voltage_index = data["feature_names"].index("vm_pu")
                scores[task]["cmllm_voltage_violation"].append(
                    radial_voltage_violation_rate(
                        predicted_real[..., voltage_index], data["sensor_edges"]
                    )
                )
    summary = {
        task: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for task, metrics in scores.items()
    }
    output = Path(args.checkpoint).parent / "evaluation.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(output)


if __name__ == "__main__":
    main()
