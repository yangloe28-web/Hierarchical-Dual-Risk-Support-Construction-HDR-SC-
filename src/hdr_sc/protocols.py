from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable
import numpy as np


@runtime_checkable
class DetectorAdapter(Protocol):
    """Minimal detector interface required by HDR-SC."""

    name: str

    def prepare(self, pool_paths: Sequence[str]) -> None:
        """Prepare or cache reusable features for the complete normal pool."""

    def fit(self, support_indices: tuple[int, ...]) -> None:
        """Build the detector's normal representation from the support set."""

    def score(self, evaluation_indices: tuple[int, ...]) -> Sequence[float]:
        """Return one native image-level anomaly score per evaluation image."""

    def anomaly_map(self, evaluation_index: int) -> np.ndarray:
        """Return one native nonnegative H x W normal-image map, not a batch.

        Run inference without gradients; detach to CPU and release GPU tensors
        before returning. Do not retain the map in the adapter.
        """

