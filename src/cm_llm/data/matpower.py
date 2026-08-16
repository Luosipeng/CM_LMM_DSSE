"""Subprocess bridge to the local MATLAB/MATPOWER installation."""

from __future__ import annotations

import subprocess
from pathlib import Path

from cm_llm.config import PROJECT_ROOT


def _matlab_quote(value: str | Path) -> str:
    return str(value).replace("'", "''").replace("\\", "/")


def run_matpower_generation(
    output_path: str | Path,
    matpower_root: str | Path,
    n_steps: int,
    seed: int,
    sample_minutes: float,
    matlab_executable: str = "matlab",
) -> Path:
    """Run the case33bw MATLAB generator and return the generated MAT path.

    Each time step is produced by a full nonlinear AC power-flow solve. The
    function deliberately does not provide a Python fallback, because doing so
    would violate the experiment's requirement that samples come from MATPOWER.
    """
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    matlab_dir = PROJECT_ROOT / "matlab"
    expression = (
        f"addpath('{_matlab_quote(matlab_dir)}'); "
        "generate_case33bw_timeseries("
        f"'{_matlab_quote(output)}',{int(n_steps)},{int(seed)},"
        f"{float(sample_minutes)},'{_matlab_quote(matpower_root)}');"
    )
    completed = subprocess.run(
        [matlab_executable, "-batch", expression],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"MATLAB/MATPOWER generation failed: {details}")
    if not output.exists():
        raise RuntimeError(f"MATLAB completed without creating {output}")
    return output

