"""Public API for Hierarchical Dual-Risk Support Construction (HDR-SC)."""

from .adapters import CallableDetectorAdapter
from .config import HDRSCConfig
from .protocols import DetectorAdapter
from .risk import RiskBreakdown, normal_risk, spatial_dispersion
from .selector import CandidateResult, HDRSCSelector, SelectionResult

__all__ = [
    "CandidateResult",
    "CallableDetectorAdapter",
    "DetectorAdapter",
    "HDRSCConfig",
    "HDRSCSelector",
    "RiskBreakdown",
    "SelectionResult",
    "normal_risk",
    "spatial_dispersion",
]
