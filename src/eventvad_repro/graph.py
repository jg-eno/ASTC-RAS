"""Paper-reconstruction-v1 graph propagation, with a dense reference implementation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .features import l2_normalize, orthogonal_projection


@dataclass(frozen=True)
class GraphConfig:
    block_size: int = 200
    clip_weight: float = .75
    time_decay: float = .6
    seed: int = 42
    qk_dim: int = 64


def build_blockwise_graph(features: np.ndarray, config: GraphConfig = GraphConfig()) -> list[tuple[np.ndarray, np.ndarray]]:
    """Released blockwise topology, repaired for N<10 and represented compactly.

    Each return item contains neighbor indices and positive Eq.4 edge weights.
    """
    values = np.asarray(features, dtype=np.float32)
    n = len(values)
    if values.ndim != 2 or values.shape[1] != 640:
        raise ValueError("features must have shape [N, 640]")
    clip, motion = l2_normalize(values[:, :512]), values[:, 512:]
    edges: list[dict[int, float]] = [dict() for _ in range(n)]
    denominator = max(1, n // 10)
    for left in range(0, n, config.block_size):
        left_end = min(n, left + config.block_size)
        positions = np.arange(left, left_end)
        dynamic_k = max(3, 5 - (left // denominator))
        for right in range(0, n, config.block_size):
            right_end = min(n, right + config.block_size)
            other_positions = np.arange(right, right_end)
            similarity = clip[left:left_end] @ clip[right:right_end].T
            distance = np.linalg.norm(motion[left:left_end, None] - motion[None, right:right_end], axis=2)
            weights = (config.clip_weight * similarity + (1 - config.clip_weight) * np.exp(-distance))
            weights /= 1 + config.time_decay * np.abs(positions[:, None] - other_positions[None, :])
            k = min(dynamic_k, right_end - right)
            for local_i, global_i in enumerate(positions):
                selected = np.argpartition(-weights[local_i], k - 1)[:k]
                for local_j in selected:
                    global_j, weight = int(other_positions[local_j]), float(weights[local_i, local_j])
                    if global_i != global_j and weight > 0:
                        edges[global_i][global_j] = max(edges[global_i].get(global_j, -np.inf), weight)
                        edges[global_j][global_i] = max(edges[global_j].get(global_i, -np.inf), weight)
    return [(np.fromiter(edge.keys(), dtype=np.int64), np.fromiter(edge.values(), dtype=np.float32)) for edge in edges]


def propagate(features: np.ndarray, adjacency: list[tuple[np.ndarray, np.ndarray]], seed: int = 42,
              qk_dim: int = 64, iterations: int = 1) -> np.ndarray:
    """Paper Eq. (5)--(8): fixed orthogonal Q/K/V, edge-weighted attention, residual, centering."""
    values = np.asarray(features, dtype=np.float32)
    if len(values) != len(adjacency):
        raise ValueError("adjacency and feature length differ")
    if values.ndim != 2 or values.shape[1] != 640 or qk_dim < 1 or iterations < 1:
        raise ValueError("expected [N,640] features, positive qk_dim, and positive iterations")
    q = orthogonal_projection(640, qk_dim, seed)
    k = orthogonal_projection(640, qk_dim, seed + 1)
    v = orthogonal_projection(640, 640, seed + 2)
    for _ in range(iterations):
        queries, keys, values_v = values @ q, values @ k, values @ v
        updated = values.copy()
        for index, (neighbors, weights) in enumerate(adjacency):
            if len(neighbors):
                logits = queries[index] @ keys[neighbors].T / np.sqrt(qk_dim)
                logits -= logits.max()
                attention = np.exp(logits); attention /= attention.sum()
                updated[index] += (attention[:, None] * weights[:, None] * values_v[neighbors]).sum(axis=0)
        values = updated - updated.mean(axis=0, keepdims=True)
    return values.astype(np.float32)
