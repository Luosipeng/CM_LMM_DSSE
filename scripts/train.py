"""Train CM-LLM with local Qwen3.5 or run a small pipeline smoke experiment."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cm_llm.config import load_config
from cm_llm.losses import ReconstructionLoss
from cm_llm.model.cmllm import CMLLM, count_parameters, load_qwen_backbone
from cm_llm.model.tiny import TinyCausalBackbone, TinyTokenizer
from cm_llm.prompts import build_prompts
from cm_llm.training import prepare_experiment_data, save_checkpoint, seed_everything


def _make_model(config: dict, data: dict, backbone_kind: str, allow_download: bool) -> CMLLM:
    model_config = config["model"]
    if backbone_kind == "qwen":
        model_source = model_config["model_path"] or model_config["model_id"]
        processor, backbone = load_qwen_backbone(
            model_source,
            dtype=model_config["dtype"],
            use_lora=model_config["use_lora"],
            lora_rank=model_config["lora_rank"],
            lora_alpha=model_config["lora_alpha"],
            lora_dropout=model_config["lora_dropout"],
            local_files_only=not allow_download,
        )
        tokenizer = processor.tokenizer
    else:
        backbone = TinyCausalBackbone()
        tokenizer = TinyTokenizer()
        # The smoke backbone is random and trainable; it makes no claim about
        # Qwen accuracy and exists solely to exercise the end-to-end pipeline.
    return CMLLM(
        backbone,
        tokenizer,
        feature_count=len(data["feature_names"]),
        sensor_count=len(data["sensor_buses"]),
        adapter_hidden_size=model_config["hidden_size"],
        dgp_layers=model_config["dgp_layers"],
        dropout=model_config["dropout"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--backbone", choices=("qwen", "tiny"), default="qwen")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.output_dir is not None:
        config["training"]["output_dir"] = str(Path(args.output_dir).resolve())
    seed_everything(config["seed"])
    data = prepare_experiment_data(config)
    model = _make_model(config, data, args.backbone, args.allow_download)
    if args.backbone == "qwen":
        device = next(model.backbone.parameters()).device
        for module in (model.temporal, model.causal, model.to_llm, model.output_projection):
            module.to(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    inventory = count_parameters(model)
    print(json.dumps(inventory, indent=2))

    training = config["training"]
    loss_function = ReconstructionLoss(
        training["mask_loss_weight"], training["graph_temporal_weight"]
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, training["epochs"])
    )
    graph_tensors = data["graph"].to_torch(device)
    history: list[dict[str, float]] = []

    def validation_loss() -> float:
        model.eval()
        total = 0.0
        batches = 0
        with torch.inference_mode():
            for batch_index, batch in enumerate(data["val_loader"]):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                values = batch["values"].to(device)
                valid_time = batch["valid_time"].to(device)
                prompts = build_prompts(
                    values,
                    batch["tasks"],
                    data["sensor_buses"].tolist(),
                    data["feature_names"],
                )
                with (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                ):
                    prediction = model(values, graph_tensors, prompts, valid_time)
                    loss, _ = loss_function(
                        prediction,
                        batch["target"].to(device),
                        batch["mask"].to(device),
                        valid_time,
                        graph_tensors,
                    )
                total += float(loss)
                batches += 1
        return total / max(batches, 1)

    for epoch in range(training["epochs"]):
        model.train()
        totals = {"loss": 0.0, "accuracy": 0.0, "masked": 0.0, "graph": 0.0}
        batches = 0
        for batch_index, batch in enumerate(data["train_loader"]):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            values = batch["values"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            valid_time = batch["valid_time"].to(device)
            prompts = build_prompts(
                values, batch["tasks"], data["sensor_buses"].tolist(), data["feature_names"]
            )
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with autocast:
                prediction = model(values, graph_tensors, prompts, valid_time)
                loss, parts = loss_function(
                    prediction,
                    target,
                    mask,
                    valid_time,
                    graph_tensors,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for key, value in parts.items():
                totals[key] += float(value.detach())
            batches += 1
        scheduler.step()
        record = {key: value / max(batches, 1) for key, value in totals.items()}
        record.update(
            {
                "epoch": epoch + 1,
                "learning_rate": scheduler.get_last_lr()[0],
                "validation_loss": validation_loss(),
            }
        )
        history.append(record)
        print(json.dumps(record))

    checkpoint = save_checkpoint(
        training["output_dir"],
        model,
        data["graph"],
        data["standardizer"],
        config,
        history,
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
