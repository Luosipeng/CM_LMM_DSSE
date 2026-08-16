"""Structured, statistics-aware prompts corresponding to the paper appendix."""

from __future__ import annotations

import torch


TASK_TEXT = {
    "imputation": "recover the contiguous missing sensor measurements",
    "forecasting": "forecast the masked future sensor measurements",
    "super_resolution": "reconstruct intermediate high-resolution measurements",
}


def build_prompts(
    values: torch.Tensor,
    tasks: list[str],
    sensor_buses: list[int],
    feature_names: list[str],
) -> list[str]:
    """Build one fixed-format dynamic prompt per sample.

    Statistics exclude the -1 sentinel. Values are normalized, so descriptions
    stay compact and avoid leaking target values into the instruction text.
    """
    prompts: list[str] = []
    for sample, task in zip(values, tasks, strict=True):
        observed = sample[sample >= 0]
        mean = observed.mean().item() if observed.numel() else 0.0
        std = observed.std().item() if observed.numel() > 1 else 0.0
        first = sample[0][sample[0] >= 0]
        last = sample[-1][sample[-1] >= 0]
        trend = (last.mean() - first.mean()).item() if first.numel() and last.numel() else 0.0
        prompts.append(
            "You are a power-system time-series analyst. "
            f"Task: {TASK_TEXT[task]}. "
            "Data source: nonlinear AC power-flow samples from MATPOWER case33bw. "
            f"Sensor buses: {sensor_buses}. Features: {feature_names}. "
            "Sensors are spatially coupled by the radial feeder and each sensor "
            "is temporally dependent on its own history. "
            f"Observed normalized statistics: mean={mean:.4f}, std={std:.4f}, "
            f"endpoint trend={trend:.4f}. Missing entries equal -1. "
            "Reconstruct all sensor-feature channels while preserving observed context."
        )
    return prompts

