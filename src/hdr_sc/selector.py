from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .candidates import generate_candidates
from .config import HDRSCConfig
from .protocols import DetectorAdapter
from .risk import normal_risk, spatial_dispersion


@dataclass
class CandidateResult:
    candidate_id: int
    support_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]
    mean_risk: float
    tail_risk: float
    total_risk: float
    tail_count: int
    rank: int = 0
    admissible: bool = False
    spatial_risk: float | None = None
    spatial_rank: int | None = None


@dataclass
class SelectionResult:
    detector: str
    pool_paths: tuple[str, ...]
    selected: CandidateResult
    ranking: list[CandidateResult]
    q_min: float
    median_q: float
    mad_q: float
    epsilon: float
    omega_size: int
    config: HDRSCConfig


ProgressCallback = Callable[[int, int, CandidateResult], None]


class HDRSCSelector:
    """Run detector-aware normal support-set selection."""

    def __init__(self, config: HDRSCConfig):
        self.config = config

    def select(
        self,
        pool_paths: Sequence[str],
        adapter: DetectorAdapter,
        progress: ProgressCallback | None = None,
    ) -> SelectionResult:
        paths = tuple(str(path) for path in pool_paths)
        if len(paths) <= self.config.support_size:
            raise ValueError("support budget must be smaller than the normal pool")
        if len(set(paths)) != len(paths):
            raise ValueError("normal pool paths must be unique")

        adapter.prepare(paths)
        candidates = generate_candidates(
            pool_size=len(paths),
            support_size=self.config.support_size,
            max_candidates=self.config.max_candidates,
            seed=self.config.seed,
        )
        pool_indices = set(range(len(paths)))
        records: list[CandidateResult] = []

        for candidate_id, support_indices in enumerate(candidates, start=1):
            support_set = set(support_indices)
            evaluation_indices = tuple(sorted(pool_indices - support_set))
            if support_set.intersection(evaluation_indices):
                raise RuntimeError("support and evaluation indices overlap")
            if not evaluation_indices:
                raise ValueError("HDR-SC requires at least one non-support normal image")

            adapter.fit(support_indices)
            scores = np.asarray(adapter.score(evaluation_indices), dtype=np.float64)
            if scores.reshape(-1).size != len(evaluation_indices):
                raise ValueError("adapter must return one score per evaluation image")
            risk = normal_risk(
                scores,
                tail_fraction=self.config.tail_fraction,
                beta=self.config.beta,
            )
            record = CandidateResult(
                candidate_id=candidate_id,
                support_indices=support_indices,
                evaluation_indices=evaluation_indices,
                mean_risk=risk.mean_risk,
                tail_risk=risk.tail_risk,
                total_risk=risk.total_risk,
                tail_count=risk.tail_count,
            )
            records.append(record)
            if progress is not None:
                progress(candidate_id, len(candidates), record)

        records.sort(
            key=lambda row: (
                row.total_risk,
                row.candidate_id,
                row.support_indices,
            )
        )
        for rank, record in enumerate(records, start=1):
            record.rank = rank

        q_values = np.asarray([row.total_risk for row in records])
        q_min = float(q_values.min())
        median_q = float(np.median(q_values))
        mad_q = float(np.median(np.abs(q_values - median_q)))
        epsilon = self.config.epsilon_weight * mad_q
        retained = []
        for row in records:
            # Match the existing runner's absolute roundoff slack, not a
            # percentage shortlist or a test-dependent gate.
            row.admissible = row.total_risk <= q_min + epsilon + 1e-12
            if row.admissible:
                retained.append(row)

        if len(retained) == 1:
            selected = retained[0]
            selected.spatial_rank = 1
        else:
            for row in retained:
                adapter.fit(row.support_indices)
                scalars = []
                for index in row.evaluation_indices:
                    anomaly_map = adapter.anomaly_map(index)
                    try:
                        scalars.append(spatial_dispersion(anomaly_map))
                    finally:
                        del anomaly_map
                # Spatial aggregation has fixed equal weights, independent
                # of the image-risk beta and tail configuration.
                row.spatial_risk = normal_risk(scalars, 0.10, 1.0).total_risk
            retained.sort(key=lambda row: (row.spatial_risk, row.candidate_id))
            for rank, row in enumerate(retained, start=1):
                row.spatial_rank = rank
            selected = retained[0]
        adapter.fit(selected.support_indices)
        return SelectionResult(
            detector=str(getattr(adapter, "name", adapter.__class__.__name__)),
            pool_paths=paths,
            selected=selected,
            ranking=records,
            q_min=q_min,
            median_q=median_q,
            mad_q=mad_q,
            epsilon=epsilon,
            omega_size=len(retained),
            config=self.config,
        )

