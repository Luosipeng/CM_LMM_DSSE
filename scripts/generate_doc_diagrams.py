"""Generate documentation diagrams without requiring Mermaid or Graphviz."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"


def _box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    color: str,
) -> None:
    """Draw one fixed-size process box in normalized axis coordinates."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.4,
        edgecolor="#25313c",
        facecolor=color,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * 0.67,
        title,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#17212b",
    )
    axis.text(
        x + width / 2,
        y + height * 0.31,
        detail,
        ha="center",
        va="center",
        fontsize=8.2,
        color="#25313c",
        linespacing=1.25,
    )


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str = "",
) -> None:
    """Draw a directed connector and an optional label."""
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.3,
            color="#3b4854",
            connectionstyle="arc3,rad=0",
        )
    )
    if label:
        axis.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.018,
            label,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#3b4854",
        )


def _elbow_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    via_y: float,
    label: str = "",
) -> None:
    """Route a connector below boxes so it does not cross their contents."""
    axis.plot(
        [start[0], start[0], end[0]],
        [start[1], via_y, via_y],
        linewidth=1.3,
        color="#3b4854",
    )
    axis.add_patch(
        FancyArrowPatch(
            (end[0], via_y),
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.3,
            color="#3b4854",
        )
    )
    if label:
        axis.text(
            (start[0] + end[0]) / 2,
            via_y + 0.012,
            label,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#3b4854",
        )


def generate_framework() -> Path:
    """Create the end-to-end data, graph, model, and optimization diagram."""
    figure, axis = plt.subplots(figsize=(16, 9), dpi=160)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.patch.set_facecolor("#f7f9fb")
    axis.set_facecolor("#f7f9fb")

    axis.text(
        0.5,
        0.96,
        "CM-LLM Reproduction: End-to-End Framework",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#17212b",
    )

    width, height = 0.17, 0.12
    top_y, mid_y, low_y = 0.73, 0.48, 0.21
    xs = [0.025, 0.22, 0.415, 0.61, 0.805]

    _box(axis, xs[0], top_y, width, height, "MATPOWER case33bw", "2016 AC power-flow states\n10 sensors x 6 variables", "#d8ecf7")
    _box(axis, xs[1], top_y, width, height, "Leakage-safe data", "chronological 70/15/15 split\ntrain-only min-max scaling", "#d8ecf7")
    _box(axis, xs[2], top_y, width, height, "Unified tasks", "imputation / forecasting / SR\nmasked value sentinel = -1", "#d8ecf7")
    _box(axis, xs[3], top_y, width, height, "Numeric tensor", "X_tilde, target, mask\n[B,T,N,F]", "#d8ecf7")
    _box(axis, xs[4], top_y, width, height, "Prompt builder", "task + buses + feature names\nobserved summary only", "#fae5c8")

    _box(axis, xs[0], mid_y, width, height, "Block-sparse VAR", "electrical candidates\nB[e,lag,F,F]", "#dcefd8")
    _box(axis, xs[1], mid_y, width, height, "ST embedding", "value + sensor + time\n[B,T,N,H_a]", "#dcefd8")
    _box(axis, xs[2], mid_y, width, height, "Sparse DGP", "ancestor + descendant\n+ variable-block messages", "#dcefd8")
    _box(axis, xs[3], mid_y, width, height, "Fusion and projection", "E_ts + E_graph -> H_q\nflatten to [B,TN,H_q]", "#dcefd8")
    _box(axis, xs[4], mid_y, width, height, "Tokenizer embedding", "left-padded prompt tokens\n[B,L_txt,H_q]", "#fae5c8")

    _box(axis, xs[0], low_y, width, height, "Loss", "L_acc + lambda_m L_mask\n+ lambda_G L_graph", "#eadff3")
    _box(axis, xs[1], low_y, width, height, "Output projection", "slice last TN states\nreshape [B,T,N,F]", "#eadff3")
    _box(axis, xs[2], low_y, width, height, "Frozen Qwen3.5", "inputs_embeds + attention mask\nbase weights unchanged", "#eadff3")
    _box(axis, xs[3], low_y, width, height, "LoRA adapters", "W = W0 + (alpha/r)BA\nonly small matrices train", "#eadff3")
    _box(axis, xs[4], low_y, width, height, "Adapter checkpoint", "LoRA + ST + DGP\n+ input/output projections", "#eadff3")

    for left, right in zip(xs[:-1], xs[1:]):
        _arrow(axis, (left + width, top_y + height / 2), (right, top_y + height / 2))
    _arrow(axis, (xs[0] + width / 2, top_y), (xs[0] + width / 2, mid_y + height), "train split")
    _arrow(axis, (xs[3] + width / 2, top_y), (xs[1] + width / 2, mid_y + height), "values")
    _arrow(axis, (xs[1] + width, mid_y + height / 2), (xs[2], mid_y + height / 2))
    _elbow_arrow(
        axis,
        (xs[0] + width / 2, mid_y),
        (xs[2] + width / 2, mid_y),
        mid_y - 0.055,
        "sparse graph",
    )
    _arrow(axis, (xs[2] + width, mid_y + height / 2), (xs[3], mid_y + height / 2))
    _arrow(axis, (xs[4] + width / 2, top_y), (xs[4] + width / 2, mid_y + height))
    _arrow(axis, (xs[3] + width / 2, mid_y), (xs[2] + width / 2, low_y + height), "numeric tokens")
    _arrow(axis, (xs[4] + width / 2, mid_y), (xs[2] + width / 2, low_y + height), "prompt prefix")
    _arrow(axis, (xs[3], low_y + height / 2), (xs[2] + width, low_y + height / 2))
    _arrow(axis, (xs[2], low_y + height / 2), (xs[1] + width, low_y + height / 2))
    _arrow(axis, (xs[1], low_y + height / 2), (xs[0] + width, low_y + height / 2), "prediction")
    _arrow(axis, (xs[3] + width, low_y + height / 2), (xs[4], low_y + height / 2), "save")

    axis.text(
        0.5,
        0.08,
        "Backpropagation updates LoRA, ST embedding, DGP, and projections; Qwen base weights stay frozen.\n"
        "The six physical variables remain separate throughout graph estimation and reconstruction.",
        ha="center",
        va="center",
        fontsize=10,
        color="#25313c",
    )
    output = FIGURE_DIR / "framework_overview.png"
    figure.savefig(output, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return output


def generate_tensor_flow() -> Path:
    """Create a shape-annotated prompt/numeric/Qwen sequence diagram."""
    figure, axis = plt.subplots(figsize=(16, 8), dpi=160)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.patch.set_facecolor("#f7f9fb")
    axis.set_facecolor("#f7f9fb")
    axis.text(
        0.5,
        0.95,
        "Exact Tensor Construction, Concatenation, and Projection",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#17212b",
    )

    width, height = 0.18, 0.125
    x1, x2, x3, x4 = 0.04, 0.29, 0.55, 0.79
    y_top, y_mid, y_low = 0.70, 0.43, 0.16

    _box(axis, x1, y_top, width, height, "Prompt strings", "list[str], length B\none task-aware prompt/sample", "#fae5c8")
    _box(axis, x2, y_top, width, height, "Tokenizer", "input_ids, attention_mask\n[B,L_txt] (left padded)", "#fae5c8")
    _box(axis, x3, y_top, width, height, "Qwen embedding table", "E_txt = Embed(input_ids)\n[B,L_txt,H_q]", "#fae5c8")

    _box(axis, x1, y_mid, width, height, "Masked measurements", "X_tilde in [B,T,N,F]\nobserved in [0,1], missing=-1", "#d8ecf7")
    _box(axis, x2, y_mid, width, height, "ST + sparse DGP", "E_ts + E_graph\n[B,T,N,H_a]", "#dcefd8")
    _box(axis, x3, y_mid, width, height, "Linear + reshape", "E_num in [B,T*N,H_q]\ntime-major, sensor-minor", "#dcefd8")

    _box(axis, x4, y_mid, width, height, "Concatenate", "[E_txt ; E_num]\n[B,L_txt+T*N,H_q]", "#eadff3")
    _box(axis, x4, y_low, width, height, "Frozen Qwen + LoRA", "last hidden state\n[B,L_txt+T*N,H_q]", "#eadff3")
    _box(axis, x3, y_low, width, height, "Hidden-state slice", "H[:, -T*N:, :]\n[B,T*N,H_q]", "#eadff3")
    _box(axis, x2, y_low, width, height, "Output projection", "Linear(H_q -> F)\n[B,T*N,F]", "#eadff3")
    _box(axis, x1, y_low, width, height, "Reconstruction", "reshape to X_hat\n[B,T,N,F]", "#eadff3")

    _arrow(axis, (x1 + width, y_top + height / 2), (x2, y_top + height / 2))
    _arrow(axis, (x2 + width, y_top + height / 2), (x3, y_top + height / 2))
    _arrow(axis, (x1 + width, y_mid + height / 2), (x2, y_mid + height / 2))
    _arrow(axis, (x2 + width, y_mid + height / 2), (x3, y_mid + height / 2))
    _arrow(axis, (x3 + width, y_top + height / 2), (x4, y_mid + height * 0.72), "prefix")
    _arrow(axis, (x3 + width, y_mid + height / 2), (x4, y_mid + height * 0.28), "append")
    _arrow(axis, (x4 + width / 2, y_mid), (x4 + width / 2, y_low + height))
    _arrow(axis, (x4, y_low + height / 2), (x3 + width, y_low + height / 2))
    _arrow(axis, (x3, y_low + height / 2), (x2 + width, y_low + height / 2))
    _arrow(axis, (x2, y_low + height / 2), (x1 + width, y_low + height / 2))

    axis.text(
        0.5,
        0.055,
        "Attention mask = concat(prompt attention mask, repeat_interleave(valid_time, N)); no language-model text is generated.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#25313c",
    )
    output = FIGURE_DIR / "qwen_tensor_flow.png"
    figure.savefig(output, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return output


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in (generate_framework(), generate_tensor_flow()):
        print(path)


if __name__ == "__main__":
    main()
