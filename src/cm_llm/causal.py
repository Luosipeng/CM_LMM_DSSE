"""Scalable physics-constrained block-sparse dynamic causal discovery.

The random variable at sensor ``i`` is the full measurement vector
``x_i(t) in R^F``. No feature averaging is performed. A directed sensor edge
stores a lagged ``F x F`` coefficient block, so individual relations such as
active-power-to-voltage remain identifiable without materializing a dense
``(N*F) x (N*F)`` graph.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import torch


@dataclass(frozen=True)
class CausalGraph:
    """Sparse block dynamic graph.

    ``lag_coefficients[e, tau, a, b]`` is the signed effect from source
    feature ``a`` at lag ``tau + 1`` to target feature ``b`` on edge ``e``.
    Self edges represent local multivariate temporal dynamics. Spatial edge
    count is therefore O(N*d), not O(N^2), when candidate degree is bounded.
    """

    edge_index: np.ndarray
    edge_weights: np.ndarray
    lag_coefficients: np.ndarray
    edge_types: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    explicit_edges: tuple[tuple[int, int], ...]
    node_count: int
    feature_count: int
    lag_order: int

    @property
    def adjacency(self) -> np.ndarray:
        """Materialize a sensor-level matrix for reporting only."""
        result = np.zeros((self.node_count, self.node_count), dtype=np.float32)
        for edge, (source, target) in enumerate(self.edge_index.T):
            if source != target:
                result[source, target] = self.edge_weights[edge]
        return result

    @property
    def spatial_edge_count(self) -> int:
        return int(np.sum(self.edge_index[0] != self.edge_index[1]))

    @property
    def variable_relation_count(self) -> int:
        return int(np.sum(np.abs(self.lag_coefficients) > 1e-8))

    def to_torch(
        self,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Create the sparse tensor representation consumed by DGP and loss."""
        return {
            "edge_index": torch.as_tensor(self.edge_index, dtype=torch.long, device=device),
            "edge_weights": torch.as_tensor(
                self.edge_weights, dtype=dtype, device=device
            ),
            "lag_coefficients": torch.as_tensor(
                self.lag_coefficients, dtype=dtype, device=device
            ),
            "feature_mean": torch.as_tensor(
                self.feature_mean, dtype=dtype, device=device
            ),
            "feature_scale": torch.as_tensor(
                self.feature_scale, dtype=dtype, device=device
            ),
        }


def _feature_kind(name: str) -> str:
    name = name.lower()
    if "vm" in name or "voltage_magnitude" in name:
        return "voltage"
    if "va" in name or "angle" in name:
        return "angle"
    if "p_inj" in name or "active_injection" in name:
        return "p_injection"
    if "q_inj" in name or "reactive_injection" in name:
        return "q_injection"
    if "p_" in name or "active" in name:
        return "p_flow"
    if "q_" in name or "reactive" in name:
        return "q_flow"
    return "other"


def physics_feature_mask(feature_names: list[str]) -> np.ndarray:
    """Return source-feature x target-feature relations allowed by grid physics.

    Constant-power injections are treated as exogenous: remote voltage and flow
    do not cause them. Voltage, angle and branch-flow targets may depend on all
    upstream/downstream electrical variables. Unknown feature types remain
    unrestricted rather than being silently discarded.
    """
    kinds = [_feature_kind(name) for name in feature_names]
    count = len(kinds)
    mask = np.ones((count, count), dtype=np.float64)
    for target, target_kind in enumerate(kinds):
        if target_kind in {"p_injection", "q_injection"}:
            for source, source_kind in enumerate(kinds):
                if source_kind not in {"p_injection", "q_injection", "other"}:
                    mask[source, target] = 0.0
    return mask


def _candidate_sources(
    node_count: int,
    explicit_edges: set[tuple[int, int]],
    candidate_hops: int,
    max_neighbors: int,
    electrical_distance: np.ndarray | None,
) -> dict[int, list[int]]:
    """Build bounded electrical neighborhoods, always retaining physical edges."""
    topology = nx.Graph()
    topology.add_nodes_from(range(node_count))
    topology.add_edges_from((source, target) for source, target in explicit_edges)
    result: dict[int, list[int]] = {}
    for target in range(node_count):
        mandatory = {
            source for source, physical_target in explicit_edges if physical_target == target
        }
        mandatory.update(
            physical_target
            for source, physical_target in explicit_edges
            if source == target
        )
        if electrical_distance is not None:
            ranked = [
                index
                for index in np.argsort(electrical_distance[:, target])
                if index != target and np.isfinite(electrical_distance[index, target])
            ]
        else:
            lengths = nx.single_source_shortest_path_length(
                topology, target, cutoff=max(1, candidate_hops)
            )
            ranked = [
                node
                for node, _ in sorted(lengths.items(), key=lambda item: (item[1], item[0]))
                if node != target
            ]
        selected = list(sorted(mandatory))
        for source in ranked:
            if source not in selected and len(selected) < max_neighbors:
                selected.append(source)
        result[target] = [target] + selected
    return result


def _largest_gram_eigenvalue(design: np.ndarray, iterations: int = 30) -> float:
    """Power iteration for ||X||_2^2/n without a dense SVD."""
    columns = design.shape[1]
    vector = np.full(columns, 1.0 / np.sqrt(max(columns, 1)))
    for _ in range(iterations):
        updated = design.T @ (design @ vector) / max(len(design), 1)
        norm = np.linalg.norm(updated)
        if norm < 1e-12:
            return 1.0
        vector = updated / norm
    eigenvalue = vector @ (design.T @ (design @ vector)) / max(len(design), 1)
    return max(float(eigenvalue), 1e-8)


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _fit_target_block_var(
    standardized: np.ndarray,
    target: int,
    sources: list[int],
    explicit_edges: set[tuple[int, int]],
    feature_mask: np.ndarray,
    *,
    lag_order: int,
    group_lasso: float,
    l1_penalty: float,
    physical_penalty_multiplier: float,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    """Fit one target with proximal sparse-group multivariate regression.

    Mathematical objective:

      min_B ||Y-XB||_F^2/(2T)
            + lambda_g sum_s omega_s ||B_s||_F
            + lambda_1 ||B||_1.

    Each group ``B_s`` contains all lags and variable pairs from one source
    sensor. Targets are independent conditional regressions and can therefore
    be parallelized for large feeders.
    """
    time_steps, _, feature_count = standardized.shape
    if time_steps <= lag_order:
        raise ValueError("Time series is shorter than the requested VAR lag order")
    source_blocks = []
    for source in sources:
        lagged = [
            standardized[lag_order - lag : time_steps - lag, source, :]
            for lag in range(1, lag_order + 1)
        ]
        source_blocks.append(np.concatenate(lagged, axis=1))
    design = np.concatenate(source_blocks, axis=1)
    response = standardized[lag_order:, target, :]
    coefficients = np.zeros((design.shape[1], feature_count), dtype=np.float64)
    step = 1.0 / _largest_gram_eigenvalue(design)

    masks: list[np.ndarray] = []
    penalties: list[float] = []
    for source in sources:
        relation_mask = np.ones_like(feature_mask) if source == target else feature_mask
        masks.append(np.tile(relation_mask, (lag_order, 1)))
        if source == target:
            penalties.append(0.0)
        elif (source, target) in explicit_edges:
            penalties.append(physical_penalty_multiplier)
        else:
            penalties.append(1.0)

    block_width = lag_order * feature_count
    for _ in range(max_iterations):
        previous = coefficients.copy()
        gradient = design.T @ (design @ coefficients - response) / len(design)
        coefficients -= step * gradient
        for group, (mask, penalty) in enumerate(zip(masks, penalties, strict=True)):
            start = group * block_width
            stop = start + block_width
            block = coefficients[start:stop] * mask
            block = _soft_threshold(block, step * l1_penalty)
            norm = np.linalg.norm(block)
            shrink = max(0.0, 1.0 - step * group_lasso * penalty / max(norm, 1e-12))
            coefficients[start:stop] = block * shrink
        relative_change = np.linalg.norm(coefficients - previous) / max(
            np.linalg.norm(previous), 1.0
        )
        if relative_change < tolerance:
            break

    result = np.zeros(
        (len(sources), lag_order, feature_count, feature_count), dtype=np.float32
    )
    for group in range(len(sources)):
        start = group * block_width
        stop = start + block_width
        result[group] = coefficients[start:stop].reshape(
            lag_order, feature_count, feature_count
        )
    return result


def discover_causal_graph(
    values: np.ndarray,
    explicit_edges: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    electrical_distance: np.ndarray | None = None,
    lag_order: int = 2,
    candidate_hops: int = 2,
    max_neighbors: int = 8,
    group_lasso: float = 0.02,
    l1_penalty: float = 0.002,
    physical_penalty_multiplier: float = 0.1,
    edge_threshold: float = 1e-3,
    min_edge_weight: float = 0.05,
    max_iterations: int = 400,
    tolerance: float = 1e-5,
    n_jobs: int = 1,
) -> CausalGraph:
    """Estimate a bounded-degree block-sparse dynamic graph without averaging.

    Complexity is O(T*N*d*p*F^2) and storage is O(N*d*p*F^2), where ``d`` is
    bounded by ``max_neighbors``. This is linear in sensor count for fixed
    feature count, lag order and electrical neighborhood size.
    """
    if values.ndim != 3:
        raise ValueError("Expected values with shape [time, sensor, feature]")
    _, node_count, feature_count = values.shape
    names = feature_names or [f"feature_{index}" for index in range(feature_count)]
    if len(names) != feature_count:
        raise ValueError("feature_names length does not match the data")
    feature_mean = np.mean(values, axis=0)
    feature_scale = np.maximum(np.std(values, axis=0), 1e-6)
    standardized = (values - feature_mean[None, :, :]) / feature_scale[None, :, :]
    explicit = {tuple(map(int, edge)) for edge in np.asarray(explicit_edges)}
    candidates = _candidate_sources(
        node_count,
        explicit,
        candidate_hops,
        max_neighbors,
        electrical_distance,
    )
    relation_mask = physics_feature_mask(names)

    def fit_target(target: int) -> tuple[int, list[int], np.ndarray]:
        sources = candidates[target]
        coefficients = _fit_target_block_var(
            standardized,
            target,
            sources,
            explicit,
            relation_mask,
            lag_order=lag_order,
            group_lasso=group_lasso,
            l1_penalty=l1_penalty,
            physical_penalty_multiplier=physical_penalty_multiplier,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        return target, sources, coefficients

    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            fitted_targets = list(executor.map(fit_target, range(node_count)))
    else:
        fitted_targets = [fit_target(target) for target in range(node_count)]

    selected_edges: list[tuple[int, int]] = []
    selected_coefficients: list[np.ndarray] = []
    selected_types: list[str] = []
    raw_weights: list[float] = []
    for target, sources, target_coefficients in fitted_targets:
        for source_index, source in enumerate(sources):
            block = target_coefficients[source_index]
            block_norm = float(np.linalg.norm(block))
            edge_type = (
                "self"
                if source == target
                else "physical"
                if (source, target) in explicit
                else "statistical"
            )
            mandatory = edge_type in {"self", "physical"}
            if mandatory or block_norm >= edge_threshold:
                selected_edges.append((source, target))
                selected_coefficients.append(block)
                selected_types.append(edge_type)
                raw_weights.append(block_norm)

    if not selected_edges:
        raise RuntimeError("Block-sparse VAR removed every edge")
    spatial_norms = [
        weight
        for weight, (source, target) in zip(raw_weights, selected_edges, strict=True)
        if source != target
    ]
    normalizer = max(max(spatial_norms, default=1.0), 1e-8)
    edge_weights = np.asarray(
        [
            1.0
            if source == target
            else max(min_edge_weight, min(1.0, weight / normalizer))
            for weight, (source, target) in zip(
                raw_weights, selected_edges, strict=True
            )
        ],
        dtype=np.float32,
    )
    return CausalGraph(
        edge_index=np.asarray(selected_edges, dtype=np.int64).T,
        edge_weights=edge_weights,
        lag_coefficients=np.stack(selected_coefficients).astype(np.float32),
        edge_types=tuple(selected_types),
        feature_mean=feature_mean.astype(np.float32),
        feature_scale=feature_scale.astype(np.float32),
        explicit_edges=tuple(sorted(explicit)),
        node_count=node_count,
        feature_count=feature_count,
        lag_order=lag_order,
    )


def graph_summary(graph: CausalGraph) -> dict[str, Any]:
    """Compact diagnostics for logs, tests and technical reports."""
    counts = {kind: graph.edge_types.count(kind) for kind in set(graph.edge_types)}
    return {
        "nodes": graph.node_count,
        "features_per_node": graph.feature_count,
        "lag_order": graph.lag_order,
        "spatial_edges": graph.spatial_edge_count,
        "variable_relations": graph.variable_relation_count,
        "edge_types": counts,
        "dense_expansion_nodes": graph.node_count * graph.feature_count,
        "stored_block_coefficients": int(graph.lag_coefficients.size),
    }
