from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskBreakdown:
    mean_risk: float
    tail_risk: float
    total_risk: float
    tail_count: int


def normal_risk(
    scores: np.ndarray | list[float] | tuple[float, ...],
    tail_fraction: float = 0.10,
    beta: float = 1.0,
) -> RiskBreakdown:
    """Compute mean plus upper-tail normal risk."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("normal-risk evaluation requires at least one score")
    if not np.all(np.isfinite(values)):
        raise ValueError("normal-risk scores must be finite")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be non-negative")

    tail_count = max(1, int(math.ceil(values.size * tail_fraction)))
    tail_values = np.partition(values, values.size - tail_count)[-tail_count:]
    mean_risk = float(np.mean(values))
    tail_risk = float(np.mean(tail_values))
    total_risk = mean_risk + float(beta) * tail_risk
    return RiskBreakdown(mean_risk, tail_risk, total_risk, tail_count)


def spatial_dispersion(anomaly_map: np.ndarray) -> float:
    """Reduce one native nonnegative H x W map to mean / top-1% mean.

    The caller must discard the map immediately; no maps are retained here.
    The denominator floor matches the current implementation, including the
    all-zero map convention (risk = 0).
    """
    values = np.asarray(anomaly_map, dtype=np.float64)
    if values.ndim != 2 or not values.size:
        raise ValueError("anomaly_map must be a nonempty H x W array")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("anomaly_map must be finite and nonnegative")
    values = values.reshape(-1)
    count = max(1, math.ceil(0.01 * values.size))
    top_mean = float(np.partition(values, values.size - count)[-count:].mean())
    return float(values.mean()) / max(top_mean, 1e-12)
