"""Training utilities shared by Qwen and lightweight pipeline verification."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .causal import CausalGraph, discover_causal_graph
from .data.dataset import (
    PowerTimeSeriesDataset,
    SensorStandardizer,
    dynamic_collate,
    load_case33bw_mat,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_experiment_data(config: dict[str, Any]) -> dict[str, Any]:
    """Create chronological splits, train-only normalization, graph, and loaders."""
    data = load_case33bw_mat(config["data"]["output_path"])
    values = data["values"]
    train_end = int(len(values) * config["data"]["train_ratio"])
    val_end = int(
        len(values)
        * (config["data"]["train_ratio"] + config["data"]["val_ratio"])
    )
    train_values = values[:train_end]
    val_values = values[train_end:val_end]
    test_values = values[val_end:]
    standardizer = SensorStandardizer().fit(train_values)
    causal_config = config["causal"]
    graph = discover_causal_graph(
        standardizer.transform(train_values),
        data["sensor_edges"],
        feature_names=data["feature_names"],
        electrical_distance=data["sensor_electrical_distance"],
        **causal_config,
    )
    training = config["training"]
    mask_options = {
        "imputation_mean": training["imputation_mean"],
        "imputation_std": training["imputation_std"],
        "forecast_horizon": training["forecast_horizon"],
        "super_resolution_factor": training["super_resolution_factor"],
    }

    def dataset(split_values: np.ndarray, tasks: list[str], seed_offset: int) -> PowerTimeSeriesDataset:
        return PowerTimeSeriesDataset(
            split_values,
            tasks,
            config["data"]["window"],
            config["data"]["stride"],
            standardizer,
            config["seed"] + seed_offset,
            mask_options,
        )

    train_dataset = dataset(train_values, training["tasks"], 0)
    val_dataset = dataset(val_values, training["tasks"], 10_000)
    test_dataset = dataset(test_values, training["tasks"], 20_000)
    common = {
        "batch_size": training["batch_size"],
        "collate_fn": dynamic_collate,
        "num_workers": 0,
    }
    return {
        **data,
        "standardizer": standardizer,
        "graph": graph,
        "train_loader": DataLoader(train_dataset, shuffle=True, **common),
        "val_loader": DataLoader(val_dataset, shuffle=False, **common),
        "test_loader": DataLoader(test_dataset, shuffle=False, **common),
    }


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Serialize only learned adapters, projections, DGP and LoRA matrices."""
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def save_checkpoint(
    output_dir: str | Path,
    model: torch.nn.Module,
    graph: CausalGraph,
    standardizer: SensorStandardizer,
    config: dict[str, Any],
    history: list[dict[str, float]],
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "adapter_checkpoint.pt"
    torch.save(
        {
            "trainable_state": trainable_state_dict(model),
            "adjacency": graph.adjacency,
            "edge_index": graph.edge_index,
            "edge_weights": graph.edge_weights,
            "lag_coefficients": graph.lag_coefficients,
            "edge_types": graph.edge_types,
            "graph_feature_mean": graph.feature_mean,
            "graph_feature_scale": graph.feature_scale,
            "normalizer_minimum": standardizer.minimum,
            "normalizer_scale": standardizer.scale,
            "config": config,
        },
        checkpoint,
    )
    with (output / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return checkpoint
