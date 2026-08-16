"""Leakage-safe loading and batching for case33bw sensor sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from .masking import apply_mask, build_mask


def _matlab_strings(array: np.ndarray) -> list[str]:
    return [str(item[0] if isinstance(item, np.ndarray) else item).strip() for item in array.ravel()]


@dataclass
class SensorStandardizer:
    """Per sensor-feature min-max transform fitted only on training time steps."""

    minimum: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "SensorStandardizer":
        self.minimum = np.nanmin(values, axis=0, keepdims=True)
        maximum = np.nanmax(values, axis=0, keepdims=True)
        self.scale = np.maximum(maximum - self.minimum, 1e-8)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.minimum is None or self.scale is None:
            raise RuntimeError("Standardizer must be fitted before transform")
        return np.clip((values - self.minimum) / self.scale, 0.0, 1.0).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.minimum is None or self.scale is None:
            raise RuntimeError("Standardizer must be fitted before inverse_transform")
        return values * self.scale + self.minimum


def load_case33bw_mat(path: str | Path) -> dict[str, Any]:
    """Load generated MATPOWER data and remove non-converged operating points."""
    raw = loadmat(path, squeeze_me=False)
    converged = raw["converged"].astype(bool).ravel()
    values = np.asarray(raw["sensor_values"], dtype=np.float64)[converged]
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Generated sensor data contains invalid values")
    electrical_distance = raw.get("sensor_electrical_distance")
    return {
        "values": values,
        "sensor_buses": raw["sensor_buses"].astype(int).ravel(),
        "sensor_edges": raw["sensor_edges"].astype(int) - 1,
        "sensor_electrical_distance": (
            np.asarray(electrical_distance, dtype=np.float64)
            if electrical_distance is not None
            else None
        ),
        "feature_names": _matlab_strings(raw["feature_names"]),
        "time_minutes": raw["time_minutes"].ravel()[converged],
        "full_bus_vm": np.asarray(raw["full_bus_vm"])[converged],
    }


class PowerTimeSeriesDataset(Dataset[dict[str, Any]]):
    """Sliding windows for heterogeneous mask-and-reconstruct tasks.

    A sample is a four-dimensional physical object: time x sensor x feature,
    where graph edges couple sensors in space and window order couples each
    sensor's measurements in time.
    """

    def __init__(
        self,
        values: np.ndarray,
        tasks: Sequence[str],
        window: int,
        stride: int,
        standardizer: SensorStandardizer,
        seed: int = 2026,
        mask_options: dict[str, Any] | None = None,
    ) -> None:
        if len(values) < window:
            raise ValueError("Not enough time steps for one window")
        self.values = standardizer.transform(values)
        self.tasks = tuple(tasks)
        self.window = int(window)
        self.starts = list(range(0, len(values) - window + 1, int(stride)))
        self.seed = int(seed)
        self.mask_options = mask_options or {}

    def __len__(self) -> int:
        return len(self.starts) * len(self.tasks)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window_index, task_index = divmod(index, len(self.tasks))
        task = self.tasks[task_index]
        start = self.starts[window_index]
        target = self.values[start : start + self.window].copy()
        rng = np.random.default_rng(self.seed + index)
        mask = build_mask(target.shape, task, rng, **self.mask_options)
        return {
            "values": torch.from_numpy(apply_mask(target, mask)),
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask),
            "task": task,
            "valid_length": target.shape[0],
            "start": start,
        }


def dynamic_collate(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Paper-style batch padding while retaining each valid sequence length."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    max_length = max(int(item["valid_length"]) for item in samples)
    _, sensors, features = samples[0]["values"].shape
    batch_size = len(samples)
    values = torch.full((batch_size, max_length, sensors, features), -1.0)
    targets = torch.zeros((batch_size, max_length, sensors, features))
    masks = torch.zeros((batch_size, max_length, sensors, features))
    valid_time = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for row, item in enumerate(samples):
        length = int(item["valid_length"])
        values[row, :length] = item["values"]
        targets[row, :length] = item["target"]
        masks[row, :length] = item["mask"]
        valid_time[row, :length] = True
    return {
        "values": values,
        "target": targets,
        "mask": masks,
        "valid_time": valid_time,
        "valid_lengths": torch.tensor([item["valid_length"] for item in samples]),
        "tasks": [item["task"] for item in samples],
        "starts": torch.tensor([item["start"] for item in samples]),
    }
