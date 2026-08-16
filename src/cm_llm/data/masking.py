"""Unified mask-and-reconstruct task definitions from the CM-LLM paper."""

from __future__ import annotations

import numpy as np


def build_mask(
    shape: tuple[int, int, int],
    task: str,
    rng: np.random.Generator,
    *,
    imputation_mean: float = 6.0,
    imputation_std: float = 2.0,
    forecast_horizon: int = 8,
    super_resolution_factor: int = 3,
) -> np.ndarray:
    """Construct M in X_tilde=(1-M) o X + (-1)M.

    Args:
        shape: ``(time, sensors, features)``.
        task: Imputation, forecasting, or super-resolution.
        rng: Reproducible NumPy random generator.

    The implementation extends the paper's variable-wise mask to sensor-feature
    channels, while preserving its exact semantics in the temporal dimension.
    """
    time_steps, sensors, features = shape
    mask = np.zeros(shape, dtype=np.float32)
    if task == "imputation":
        for sensor in range(sensors):
            length = int(round(rng.normal(imputation_mean, imputation_std)))
            length = int(np.clip(length, 1, max(1, time_steps - 1)))
            start = int(rng.integers(0, time_steps - length + 1))
            # Real sensor outages generally affect all co-located measurements.
            mask[start : start + length, sensor, :] = 1.0
    elif task == "forecasting":
        horizon = int(np.clip(forecast_horizon, 1, time_steps - 1))
        mask[-horizon:, :, :] = 1.0
    elif task == "super_resolution":
        factor = max(2, int(super_resolution_factor))
        observed = np.arange(0, time_steps, factor)
        mask[:, :, :] = 1.0
        mask[observed, :, :] = 0.0
        mask[-1, :, :] = 0.0
    else:
        raise ValueError(f"Unsupported task: {task}")
    return mask


def apply_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply equation (6) from the paper with -1 as the missing sentinel."""
    if values.shape != mask.shape:
        raise ValueError(f"Shape mismatch: values={values.shape}, mask={mask.shape}")
    return (1.0 - mask) * values - mask

