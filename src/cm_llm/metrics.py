"""Pointwise and physical diagnostics for reconstruction experiments."""

from __future__ import annotations

import numpy as np


def reconstruction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    selected = np.ones_like(target, dtype=bool) if mask is None else mask.astype(bool)
    error = prediction[selected] - target[selected]
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def radial_voltage_violation_rate(
    voltage: np.ndarray,
    sensor_edges: np.ndarray,
    tolerance: float = 2e-3,
) -> float:
    """Fraction of forward-feeder edges with implausible downstream voltage rise."""
    violations = []
    for source, target in np.asarray(sensor_edges, dtype=int):
        violations.append(voltage[..., target] > voltage[..., source] + tolerance)
    if not violations:
        return 0.0
    return float(np.mean(np.stack(violations)))
