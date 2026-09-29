from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HDRSCConfig:
    """Configuration for one HDR-SC support-selection task."""

    support_size: int
    max_candidates: int = 500
    tail_fraction: float = 0.10
    beta: float = 1.0
    seed: int = 0
    epsilon_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.support_size < 1:
            raise ValueError("support_size must be at least 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        if not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")
        if not math.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("beta must be non-negative")
        if not math.isfinite(self.epsilon_weight) or self.epsilon_weight < 0.0:
            raise ValueError("epsilon_weight must be finite and non-negative")

