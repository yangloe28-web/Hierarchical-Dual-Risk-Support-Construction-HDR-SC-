from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
import numpy as np


PrepareFunction = Callable[[Sequence[str]], Any]
FitFunction = Callable[[Any, tuple[int, ...]], Any]
ScoreFunction = Callable[[Any, Any, tuple[int, ...]], Sequence[float]]
MapFunction = Callable[[Any, Any, int], np.ndarray]


@dataclass
class CallableDetectorAdapter:
    """Connect an external detector through four plain Python callables.

    ``prepare_pool`` returns support-independent cached state. ``fit_support``
    receives that state and returns a support-conditioned detector state.
    ``score_normal`` returns one image-level score per evaluation index.
    """

    name: str
    prepare_pool: PrepareFunction
    fit_support: FitFunction
    score_normal: ScoreFunction
    map_normal: MapFunction
    pool_state: Any = field(default=None, init=False, repr=False)
    detector_state: Any = field(default=None, init=False, repr=False)

    def prepare(self, pool_paths: Sequence[str]) -> None:
        self.pool_state = self.prepare_pool(pool_paths)

    def fit(self, support_indices: tuple[int, ...]) -> None:
        if self.pool_state is None:
            raise RuntimeError("prepare must be called before fit")
        self.detector_state = self.fit_support(self.pool_state, support_indices)

    def score(self, evaluation_indices: tuple[int, ...]) -> Sequence[float]:
        if self.pool_state is None or self.detector_state is None:
            raise RuntimeError("fit must be called before score")
        return self.score_normal(
            self.pool_state,
            self.detector_state,
            evaluation_indices,
        )

    def anomaly_map(self, evaluation_index: int) -> np.ndarray:
        if self.pool_state is None or self.detector_state is None:
            raise RuntimeError("fit must be called before anomaly_map")
        return self.map_normal(self.pool_state, self.detector_state, evaluation_index)
