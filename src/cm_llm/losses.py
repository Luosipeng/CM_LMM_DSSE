"""Paper reconstruction objective and block dynamic-graph regularization."""

from __future__ import annotations

import torch
from torch import nn


def _masked_mean(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    selected = selected.to(values.dtype)
    return (values * selected).sum() / selected.sum().clamp_min(1.0)


def block_var_residual(
    prediction: torch.Tensor,
    graph: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, int]:
    """Evaluate the learned vector structural equation on model predictions.

    For every stored edge e=(i,j):

      z_hat_j(t) = sum_e sum_tau z_i(t-tau) B_e^(tau).

    The returned residual retains all sensor-feature dimensions; no scalar
    projection or feature mean is used.
    """
    mean = graph["feature_mean"][None, None, :, :].to(prediction.dtype)
    scale = graph["feature_scale"][None, None, :, :].to(prediction.dtype)
    standardized = (prediction - mean) / scale.clamp_min(1e-6)
    source, target = graph["edge_index"]
    coefficients = graph["lag_coefficients"].to(prediction.dtype)
    lag_order = coefficients.shape[1]
    fitted = torch.zeros_like(standardized)
    for lag in range(1, lag_order + 1):
        if lag >= prediction.shape[1]:
            break
        source_history = standardized[:, :-lag, source, :]
        edge_contribution = torch.einsum(
            "btef,efg->bteg", source_history, coefficients[:, lag - 1]
        )
        fitted[:, lag:].index_add_(2, target, edge_contribution.to(fitted.dtype))
    return standardized - fitted, lag_order


class ReconstructionLoss(nn.Module):
    """Compute L_acc + lambda_mask L_mask + lambda_g L_block_VAR.

    The first two terms reproduce equations (17)-(18) of CM-LLM. The optional
    graph term is the Gaussian negative log prior induced by the learned sparse
    vector structural equation. Setting ``graph_temporal_weight=0`` restores
    the paper reconstruction objective.
    """

    def __init__(self, mask_weight: float = 1.0, graph_temporal_weight: float = 0.0) -> None:
        super().__init__()
        self.mask_weight = float(mask_weight)
        self.graph_temporal_weight = float(graph_temporal_weight)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        valid_time: torch.Tensor,
        graph: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid = valid_time[:, :, None, None].expand_as(target)
        squared_error = (prediction - target).square()
        accuracy = _masked_mean(squared_error, valid)
        masked = _masked_mean(squared_error, valid & (mask > 0))
        graph_term = prediction.new_zeros(())
        if self.graph_temporal_weight > 0:
            if graph is None:
                raise ValueError("Block graph regularization requires graph tensors")
            prediction_residual, lag_order = block_var_residual(prediction, graph)
            with torch.no_grad():
                target_residual, _ = block_var_residual(target, graph)
            graph_valid = valid.clone()
            graph_valid[:, :lag_order] = False
            # Penalize graph-inconsistent reconstruction error rather than
            # forcing predictions to fit the imperfect VAR model more closely
            # than the real measurements do. This term is exactly zero when
            # prediction equals target.
            structural_error = prediction_residual - target_residual
            graph_term = _masked_mean(structural_error.square(), graph_valid)
        total = accuracy + self.mask_weight * masked + self.graph_temporal_weight * graph_term
        return total, {"accuracy": accuracy, "masked": masked, "graph": graph_term}
