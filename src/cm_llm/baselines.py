"""Transparent non-learned baselines for each reconstruction task."""

from __future__ import annotations

import numpy as np


def classical_reconstruction(masked: np.ndarray, task: str) -> np.ndarray:
    """Linear interpolation for in-range gaps and persistence for future gaps."""
    output = masked.copy()
    time_steps, sensors, features = output.shape
    positions = np.arange(time_steps)
    for sensor in range(sensors):
        for feature in range(features):
            series = output[:, sensor, feature]
            observed = series >= 0
            if not observed.any():
                series[:] = 0.0
                continue
            if task == "forecasting":
                first_missing = np.flatnonzero(~observed)
                if first_missing.size:
                    series[first_missing[0] :] = series[first_missing[0] - 1]
            else:
                series[~observed] = np.interp(
                    positions[~observed], positions[observed], series[observed]
                )
    return output

