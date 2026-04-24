"""
hitlo — Human-in-the-Loop exoskeleton optimization library.

Public API:
    from hitlo.detection import DetectionConfig, detect_heelstrikes_full
    from hitlo.symmetry  import compute_step_times, compute_symmetry_index
    from hitlo.cost      import SymmetryCost
    from hitlo.io        import load_both_polar_streams, trial_filename
    from hitlo.hil_exo   import HIL_Exo

Module layout:
    detection.py — heel-strike detection pipeline (single source of truth)
    symmetry.py  — step-time interleaving + symmetry index
    cost.py      — SymmetryCost (BO cost), spring-torque model
    io.py        — XDF loading, BIDS-style filename helpers
    hil_exo.py   — HIL_Exo experiment driver (wraps HIL_toolkit's BayesianOptimization
                   with exoskeleton safety constraints, LHS sampling, trial loop)
"""

__version__ = "2.0.0"

from hitlo.detection import (
    DetectionConfig,
    DetectionResult,
    compute_magnitude,
    compute_jerk_z,
    detect_peak_candidates,
    cluster_keep_last,
    detect_heelstrikes_full,
)
from hitlo.symmetry import (
    compute_step_times,
    compute_symmetry_index,
    trim_peaks,
    filter_implausible_strides,
)

__all__ = [
    "DetectionConfig",
    "DetectionResult",
    "compute_magnitude",
    "compute_jerk_z",
    "detect_peak_candidates",
    "cluster_keep_last",
    "detect_heelstrikes_full",
    "compute_step_times",
    "compute_symmetry_index",
    "trim_peaks",
    "filter_implausible_strides",
]
