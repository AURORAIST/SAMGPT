from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch


@dataclass
class StructPromptRouterConfig:
    """Configuration for structural-prototype prompt routing.

    Attributes:
        dataset_name: Downstream dataset name.
        num_prototypes: Number of structural prototypes (prompt banks).
        graph_step_id: Preferred graph id to query structural labels.
        fallback_graph_step_id: Fallback graph id if preferred id is unavailable.
        smoothing_eps: Optional additive smoothing before normalization.
    """

    dataset_name: str
    num_prototypes: int
    graph_step_id: int = 0
    fallback_graph_step_id: int = 0
    smoothing_eps: float = 0.0


class StructPrototypeWeightBuilder:
    """Convert node structural labels to prototype weight vectors."""

    @staticmethod
    def _to_numpy_index(node_idx: object) -> np.ndarray:
        if torch.is_tensor(node_idx):
            return node_idx.detach().cpu().numpy().astype(np.int64)
        if isinstance(node_idx, range):
            return np.asarray(list(node_idx), dtype=np.int64)
        return np.asarray(node_idx, dtype=np.int64)

    @staticmethod
    def build(
        node_struct_labels: Optional[np.ndarray],
        node_idx: object,
        num_prototypes: int,
        smoothing_eps: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Build normalized prototype weights from node-level cluster labels.

        Args:
            node_struct_labels: Array of shape [num_nodes], values in [0, K-1] or -1.
            node_idx: Index set for the current sample (node task or graph-subgraph nodes).
            num_prototypes: K.
            smoothing_eps: Additive smoothing before normalization.

        Returns:
            A float32 vector of shape [K] or None if unavailable.
        """
        if node_struct_labels is None or int(num_prototypes) <= 0:
            return None

        idx_np = StructPrototypeWeightBuilder._to_numpy_index(node_idx)
        if idx_np.size == 0:
            return None

        labels = np.asarray(node_struct_labels, dtype=np.int64)
        picked = labels[idx_np]
        valid = picked[(picked >= 0) & (picked < int(num_prototypes))]
        if valid.size == 0:
            return None

        counts = np.bincount(valid, minlength=int(num_prototypes)).astype(np.float32)
        eps = float(max(0.0, smoothing_eps))
        if eps > 0:
            counts = counts + eps

        denom = float(counts.sum())
        if denom < 1e-12:
            return None
        return (counts / denom).astype(np.float32)


class StructPromptRouter:
    """Task-agnostic structural prompt router.

    This class decouples "where labels come from" and "how weights are built".

    label_getter signature:
        label_getter(dataset_name: str, data_obj: object, step_id: int) -> Optional[np.ndarray]
    """

    def __init__(
        self,
        config: StructPromptRouterConfig,
        label_getter: Callable[[str, object, int], Optional[np.ndarray]],
        weight_getter: Optional[Callable[[str, object, int], Optional[np.ndarray]]] = None,
    ) -> None:
        self.config = config
        self.label_getter = label_getter
        self.weight_getter = weight_getter
        self._cached_labels: Optional[np.ndarray] = None
        self._cached_node_weights: Optional[np.ndarray] = None

    def _prepare_node_weights(self, data_obj: object) -> Optional[np.ndarray]:
        if self.weight_getter is None:
            return None

        weights = self.weight_getter(
            self.config.dataset_name,
            data_obj,
            int(self.config.graph_step_id),
        )
        if weights is None and int(self.config.fallback_graph_step_id) != int(self.config.graph_step_id):
            weights = self.weight_getter(
                self.config.dataset_name,
                data_obj,
                int(self.config.fallback_graph_step_id),
            )

        if weights is None:
            return None

        arr = np.asarray(weights, dtype=np.float32)
        if arr.ndim != 2:
            return None
        if arr.shape[1] <= 0:
            return None

        # 归一化每个节点的原型权重，避免上游数值漂移。
        row_sum = arr.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum < 1e-12, 1.0, row_sum)
        arr = arr / row_sum
        return arr

    def prepare(self, data_obj: object) -> Optional[np.ndarray]:
        """Load and cache structural node labels for a downstream graph object."""
        node_weights = self._prepare_node_weights(data_obj)
        if node_weights is not None:
            self._cached_node_weights = node_weights
            self._cached_labels = None
            return node_weights

        self._cached_node_weights = None
        labels = self.label_getter(
            self.config.dataset_name,
            data_obj,
            int(self.config.graph_step_id),
        )
        if labels is None and int(self.config.fallback_graph_step_id) != int(self.config.graph_step_id):
            labels = self.label_getter(
                self.config.dataset_name,
                data_obj,
                int(self.config.fallback_graph_step_id),
            )

        if labels is not None:
            self._cached_labels = np.asarray(labels, dtype=np.int64)
        else:
            self._cached_labels = None
        return self._cached_labels

    def weights_for(self, node_idx: object) -> Optional[np.ndarray]:
        """Build prototype weights for any task that can provide node indices."""
        if self._cached_node_weights is not None:
            idx_np = StructPrototypeWeightBuilder._to_numpy_index(node_idx)
            if idx_np.size == 0:
                return None
            picked = self._cached_node_weights[idx_np]
            if picked.size == 0:
                return None
            weights = picked.mean(axis=0).astype(np.float32)
            eps = float(max(0.0, self.config.smoothing_eps))
            if eps > 0:
                weights = weights + eps
            denom = float(weights.sum())
            if denom < 1e-12:
                return None
            return (weights / denom).astype(np.float32)

        return StructPrototypeWeightBuilder.build(
            node_struct_labels=self._cached_labels,
            node_idx=node_idx,
            num_prototypes=int(self.config.num_prototypes),
            smoothing_eps=float(self.config.smoothing_eps),
        )

    def weights_for_tensor(self, node_idx: object, device: torch.device) -> Optional[torch.Tensor]:
        """Same as weights_for but returns torch tensor on target device."""
        weights = self.weights_for(node_idx)
        if weights is None:
            return None
        return torch.as_tensor(weights, dtype=torch.float32, device=device)


__all__ = [
    "StructPromptRouterConfig",
    "StructPrototypeWeightBuilder",
    "StructPromptRouter",
]
