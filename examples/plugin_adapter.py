from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


class SyntheticNormalAdapter:
    """Tiny deterministic adapter demonstrating the HDR-SC interface."""

    name = "synthetic-normal-detector"

    def __init__(self, config: dict | None = None):
        self.dimension = int((config or {}).get("dimension", 8))
        self.features: np.ndarray | None = None
        self.center: np.ndarray | None = None

    def prepare(self, pool_paths: Sequence[str]) -> None:
        rows = []
        for path in pool_paths:
            digest = hashlib.sha256(path.encode("utf-8")).digest()
            values = np.frombuffer(digest, dtype=np.uint8)[: self.dimension]
            rows.append(values.astype(np.float64) / 255.0)
        self.features = np.stack(rows)

    def fit(self, support_indices: tuple[int, ...]) -> None:
        if self.features is None:
            raise RuntimeError("prepare must be called before fit")
        self.center = self.features[list(support_indices)].mean(axis=0)

    def score(self, evaluation_indices: tuple[int, ...]) -> np.ndarray:
        if self.features is None or self.center is None:
            raise RuntimeError("fit must be called before score")
        residual = self.features[list(evaluation_indices)] - self.center
        return np.linalg.norm(residual, axis=1)

    def anomaly_map(self, evaluation_index: int) -> np.ndarray:
        if self.features is None or self.center is None:
            raise RuntimeError("fit must be called before anomaly_map")
        # Toy residual map for interface testing only, not a real detector.
        return np.abs(self.features[evaluation_index] - self.center).reshape(1, -1)


def build_adapter(config: dict | None = None) -> SyntheticNormalAdapter:
    return SyntheticNormalAdapter(config)

