"""Load the local Qwen3.5 deployment and run one multimodal forward pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cm_llm.config import load_config
from cm_llm.model.cmllm import CMLLM, count_parameters, load_qwen_backbone


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--without-lora", action="store_true")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    source = model_config["model_path"] or model_config["model_id"]
    processor, backbone = load_qwen_backbone(
        source,
        dtype=model_config["dtype"],
        use_lora=model_config["use_lora"] and not args.without_lora,
        lora_rank=model_config["lora_rank"],
        lora_alpha=model_config["lora_alpha"],
        lora_dropout=model_config["lora_dropout"],
        local_files_only=True,
    )
    model = CMLLM(backbone, processor.tokenizer, 6, 10, 64, 1, 0.0)
    device = next(backbone.parameters()).device
    for module in (model.temporal, model.causal, model.to_llm, model.output_projection):
        module.to(device)
    values = torch.rand(1, 2, 10, 6, device=device)
    edge_index = torch.tensor(
        [list(range(10)) + list(range(9)), list(range(10)) + list(range(1, 10))],
        dtype=torch.long,
        device=device,
    )
    edge_count = edge_index.shape[1]
    causal_graph = {
        "edge_index": edge_index,
        "edge_weights": torch.ones(edge_count, device=device),
        "lag_coefficients": torch.zeros(edge_count, 2, 6, 6, device=device),
        "feature_mean": torch.full((10, 6), 0.5, device=device),
        "feature_scale": torch.full((10, 6), 0.25, device=device),
    }
    trainable_base_non_lora = [
        name
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if trainable_base_non_lora:
        raise RuntimeError(
            "Base Qwen parameters unexpectedly trainable: "
            + ", ".join(trainable_base_non_lora[:5])
        )
    if args.backward:
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(
                values,
                causal_graph,
                ["Reconstruct masked MATPOWER case33bw sensor measurements."],
            )
            loss = prediction.float().square().mean()
        loss.backward()
        lora_gradients = sum(
            parameter.grad is not None
            for name, parameter in model.backbone.named_parameters()
            if "lora_" in name
        )
    else:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(
                values,
                causal_graph,
                ["Reconstruct masked MATPOWER case33bw sensor measurements."],
            )
        lora_gradients = 0
    report = count_parameters(model)
    report.update(
        {
            "prediction_shape": list(prediction.shape),
            "prediction_finite": bool(torch.isfinite(prediction).all()),
            "device": str(device),
            "base_non_lora_trainable": len(trainable_base_non_lora),
            "lora_tensors_with_gradients": lora_gradients,
        }
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
