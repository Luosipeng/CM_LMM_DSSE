"""Sparse directed graph propagation with variable-level block relations."""

from __future__ import annotations

import torch
from torch import nn


def _row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _sparse_weighted_aggregate(
    messages: torch.Tensor,
    destination: torch.Tensor,
    weights: torch.Tensor,
    node_count: int,
) -> torch.Tensor:
    """Aggregate ``[B,T,E,H]`` messages in O(E) memory and computation."""
    output = messages.new_zeros(
        messages.shape[0], messages.shape[1], node_count, messages.shape[-1]
    )
    weighted = messages * weights[None, None, :, None].to(messages.dtype)
    output.index_add_(2, destination, weighted)
    degree = weights.new_zeros(node_count)
    degree.index_add_(0, destination, weights)
    return output / degree.clamp_min(1e-8)[None, None, :, None].to(output.dtype)


def _block_var_messages(
    values: torch.Tensor,
    graph: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Evaluate signed lagged F-by-F edge blocks on observed sensor values.

    For edge e=(i,j), the variable message is

      m_e(t) = sum_tau z_i(t-tau) B_e^(tau),

    where z is standardized with training-only graph statistics. Masked values
    are replaced by the standardization mean (zero in z-space), avoiding use of
    the numerical -1 sentinel as a physical observation.
    """
    edge_index = graph["edge_index"]
    coefficients = graph["lag_coefficients"]
    source, target = edge_index
    mean = graph["feature_mean"][None, None, :, :].to(values.dtype)
    scale = graph["feature_scale"][None, None, :, :].to(values.dtype)
    standardized = (values - mean) / scale.clamp_min(1e-6)
    standardized = torch.where(values >= 0, standardized, torch.zeros_like(standardized))
    batch, time_steps, node_count, feature_count = values.shape
    edge_count, lag_order = coefficients.shape[:2]
    edge_messages = values.new_zeros(batch, time_steps, edge_count, feature_count)
    coefficients = coefficients.to(values.dtype)
    for lag in range(1, lag_order + 1):
        if lag >= time_steps:
            break
        source_history = standardized[:, :-lag, source, :]
        contribution = torch.einsum(
            "btef,efg->bteg", source_history, coefficients[:, lag - 1]
        )
        edge_messages[:, lag:] += contribution
    node_messages = values.new_zeros(batch, time_steps, node_count, feature_count)
    node_messages.index_add_(2, target, edge_messages)
    return node_messages


class DGPBlock(nn.Module):
    """Separate ancestor, descendant and typed variable-relation propagation."""

    def __init__(self, hidden_size: int, dropout: float, feature_count: int = 0) -> None:
        super().__init__()
        self.self_projection = nn.Linear(hidden_size, hidden_size)
        self.ancestor_projection = nn.Linear(hidden_size, hidden_size)
        self.descendant_projection = nn.Linear(hidden_size, hidden_size)
        self.relation_projection = (
            nn.Linear(feature_count, hidden_size) if feature_count > 0 else None
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def _dense_messages(
        self, node_states: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0).expand(node_states.shape[0], -1, -1)
        ancestors = torch.einsum(
            "bij,btjh->btih", _row_normalize(adjacency.transpose(-1, -2)), node_states
        )
        descendants = torch.einsum(
            "bij,btjh->btih", _row_normalize(adjacency), node_states
        )
        return ancestors, descendants

    def _sparse_messages(
        self,
        node_states: torch.Tensor,
        graph: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = graph["edge_index"]
        spatial = source != target
        source = source[spatial]
        target = target[spatial]
        weights = graph["edge_weights"][spatial]
        node_count = node_states.shape[2]
        ancestors = _sparse_weighted_aggregate(
            node_states[:, :, source, :], target, weights, node_count
        )
        descendants = _sparse_weighted_aggregate(
            node_states[:, :, target, :], source, weights, node_count
        )
        return ancestors, descendants

    def forward(
        self,
        node_states: torch.Tensor,
        graph: torch.Tensor | dict[str, torch.Tensor],
        relation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(graph, torch.Tensor):
            ancestors, descendants = self._dense_messages(node_states, graph)
        else:
            ancestors, descendants = self._sparse_messages(node_states, graph)
        update = self.self_projection(node_states)
        update = update + self.ancestor_projection(ancestors)
        update = update + self.descendant_projection(descendants)
        if relation_features is not None:
            if self.relation_projection is None:
                raise RuntimeError("DGP was created without a feature relation projection")
            update = update + self.relation_projection(relation_features)
        return self.norm(node_states + self.dropout(self.activation(update)))


class DenseGraphPropagation(nn.Module):
    """Directed propagation whose sparse mode scales linearly with edge count."""

    def __init__(
        self,
        hidden_size: int,
        layers: int = 2,
        dropout: float = 0.1,
        feature_count: int = 0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DGPBlock(hidden_size, dropout, feature_count) for _ in range(layers)
        )

    def forward(
        self,
        node_states: torch.Tensor,
        graph: torch.Tensor | dict[str, torch.Tensor],
        values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        relation_features = None
        if isinstance(graph, dict):
            if values is None:
                raise ValueError("Sparse block graph propagation requires raw values")
            relation_features = _block_var_messages(values, graph)
        output = node_states
        for index, layer in enumerate(self.layers):
            output = layer(
                output,
                graph,
                relation_features if index == 0 else None,
            )
        return output
