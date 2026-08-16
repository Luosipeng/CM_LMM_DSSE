from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cm_llm.causal import discover_causal_graph, graph_summary
from cm_llm.config import load_config
from cm_llm.data.masking import apply_mask, build_mask
from cm_llm.losses import ReconstructionLoss
from cm_llm.model.cmllm import CMLLM
from cm_llm.model.dgp import DenseGraphPropagation
from cm_llm.model.tiny import TinyCausalBackbone, TinyTokenizer


class MaskingTests(unittest.TestCase):
    def test_three_task_masks_and_sentinel(self) -> None:
        values = np.full((12, 3, 2), 0.5, dtype=np.float32)
        rng = np.random.default_rng(1)
        for task in ("imputation", "forecasting", "super_resolution"):
            mask = build_mask(values.shape, task, rng, forecast_horizon=3)
            masked = apply_mask(values, mask)
            self.assertTrue(np.all(masked[mask.astype(bool)] == -1))
            self.assertGreater(mask.sum(), 0)

    def test_config_inheritance(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "improved.json")
        self.assertEqual(config["causal"]["lag_order"], 2)
        self.assertEqual(config["training"]["graph_temporal_weight"], 0.005)


class CausalTests(unittest.TestCase):
    def test_block_graph_preserves_features_and_physical_edges(self) -> None:
        rng = np.random.default_rng(2)
        values = rng.normal(size=(240, 4, 3))
        values[1:, 1, 0] += 0.8 * values[:-1, 0, 2]
        explicit = np.array([[0, 1], [1, 2], [2, 3]])
        graph = discover_causal_graph(
            values,
            explicit,
            feature_names=["vm_pu", "p_inj_mw", "q_inj_mvar"],
            lag_order=2,
            max_neighbors=3,
            max_iterations=100,
            n_jobs=2,
        )
        for source, target in explicit:
            self.assertGreater(graph.adjacency[source, target], 0)
        self.assertEqual(graph.lag_coefficients.shape[1:], (2, 3, 3))
        self.assertGreater(graph_summary(graph)["variable_relations"], 0)
        edge = np.flatnonzero(
            (graph.edge_index[0] == 0) & (graph.edge_index[1] == 1)
        )[0]
        self.assertGreater(abs(graph.lag_coefficients[edge, 0, 2, 0]), 0.5)


class NeuralTests(unittest.TestCase):
    def test_dgp_and_cmllm_shapes(self) -> None:
        values = torch.rand(2, 6, 4, 3)
        adjacency = torch.tensor(
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
            dtype=torch.float32,
        )
        dgp = DenseGraphPropagation(8, layers=2, dropout=0)
        self.assertEqual(dgp(torch.rand(2, 6, 4, 8), adjacency).shape, (2, 6, 4, 8))
        sparse_graph = {
            "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
            "edge_weights": torch.ones(3),
            "lag_coefficients": torch.zeros(3, 2, 3, 3),
            "feature_mean": torch.zeros(4, 3),
            "feature_scale": torch.ones(4, 3),
        }
        sparse_dgp = DenseGraphPropagation(8, layers=2, dropout=0, feature_count=3)
        self.assertEqual(
            sparse_dgp(torch.rand(2, 6, 4, 8), sparse_graph, values).shape,
            (2, 6, 4, 8),
        )
        model = CMLLM(
            TinyCausalBackbone(32), TinyTokenizer(), 3, 4, 16, 2, 0.0
        )
        output = model(values, sparse_graph, ["task one", "task two"])
        self.assertEqual(output.shape, values.shape)

    def test_loss_focuses_masked_region(self) -> None:
        target = torch.zeros(1, 4, 2, 1)
        prediction = target.clone()
        prediction[:, -1] = 2
        mask = torch.zeros_like(target)
        mask[:, -1] = 1
        valid = torch.ones(1, 4, dtype=torch.bool)
        loss, parts = ReconstructionLoss()(prediction, target, mask, valid)
        self.assertAlmostEqual(parts["masked"].item(), 4.0)
        self.assertGreater(loss.item(), parts["accuracy"].item())

    def test_block_graph_loss_retains_feature_dimension(self) -> None:
        prediction = torch.rand(1, 5, 2, 3, requires_grad=True)
        target = torch.rand_like(prediction)
        mask = torch.ones_like(prediction)
        valid = torch.ones(1, 5, dtype=torch.bool)
        graph = {
            "edge_index": torch.tensor([[0, 1], [0, 1]]),
            "edge_weights": torch.ones(2),
            "lag_coefficients": torch.zeros(2, 1, 3, 3),
            "feature_mean": torch.zeros(2, 3),
            "feature_scale": torch.ones(2, 3),
        }
        graph["lag_coefficients"][:, 0] = torch.eye(3)
        loss, parts = ReconstructionLoss(graph_temporal_weight=0.1)(
            prediction, target, mask, valid, graph
        )
        self.assertGreater(parts["graph"].item(), 0)
        loss.backward()
        self.assertIsNotNone(prediction.grad)


if __name__ == "__main__":
    unittest.main()
