"""
hitlo.symmetry — step-time symmetry index computation.

Given two lists of heel-strike timestamps (one per foot) on a common time
base, interleave them to compute step times, then reduce to the gait
symmetry index (SI).

Step time definitions (standard in gait rehab literature):
    Right step = LEFT heel strike  → next RIGHT heel strike
    Left step  = RIGHT heel strike → next LEFT heel strike

Stride time = one step + the subsequent step (one full gait cycle).
"""

from typing import Tuple, List
import numpy as np


# ===========================================================================
# Steady-state trim
# ===========================================================================

def trim_peaks(peak_times: np.ndarray,
               trial_start: float,
               trial_end: float,
               trim_s: float) -> np.ndarray:
    """Drop heel strikes within `trim_s` seconds of the trial start or end.

    Ramp-up and ramp-down strides have systematically different mechanics
    (weaker shank accelerations during startup) and contaminate the
    steady-state symmetry estimate. Standard practice is to trim 3-5 seconds.
    """
    if trim_s <= 0:
        return peak_times
    t_lo = trial_start + trim_s
    t_hi = trial_end - trim_s
    mask = (peak_times >= t_lo) & (peak_times <= t_hi)
    return peak_times[mask]


# ===========================================================================
# Step-time interleaving
# ===========================================================================

def compute_step_times(left_times: np.ndarray,
                       right_times: np.ndarray,
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Interleave L and R heel strikes into step times.

    IMPORTANT: left_times and right_times must be on a COMMON time base
    (real LSL timestamps). The two shank IMUs run at slightly different
    actual sample rates, so sample indices are NOT interchangeable.

    Parameters
    ----------
    left_times, right_times : ndarray (floats, LSL seconds)

    Returns
    -------
    right_step_times : ndarray  gaps L→R
    left_step_times  : ndarray  gaps R→L
    """
    all_times = np.concatenate([left_times, right_times])
    all_labels = np.array(['L'] * len(left_times) + ['R'] * len(right_times))
    order = np.argsort(all_times, kind='stable')
    all_times = all_times[order]
    all_labels = all_labels[order]

    right_step_times: List[float] = []
    left_step_times: List[float] = []

    for i in range(len(all_times) - 1):
        dt = all_times[i + 1] - all_times[i]
        if all_labels[i] == 'L' and all_labels[i + 1] == 'R':
            right_step_times.append(dt)
        elif all_labels[i] == 'R' and all_labels[i + 1] == 'L':
            left_step_times.append(dt)

    return np.array(right_step_times), np.array(left_step_times)


# ===========================================================================
# Symmetry index
# ===========================================================================

def compute_symmetry_index(right_step_times: np.ndarray,
                           left_step_times: np.ndarray,
                           signed: bool = True
                           ) -> Tuple[float, np.ndarray]:
    """Step-time symmetry index.

        SI = 2 × (right - left) / (right + left) × 100 %

    Dimensionless, bounded roughly in ±100%. Values:
        SI =  0  → perfectly symmetric
        SI > 0  → right step > left step  (left leg is support-dominant)
        SI < 0  → left step > right step  (right leg is support-dominant)

    Parameters
    ----------
    right_step_times, left_step_times : ndarray (seconds)
    signed : bool
        If True, return the mean signed SI (preserves sign of asymmetry).
        If False, return mean |SI| (magnitude only).

    Returns
    -------
    mean_si    : float                       aggregated SI across strides
    per_stride : ndarray                     SI per stride (always signed)
    """
    n = min(len(right_step_times), len(left_step_times))
    r = right_step_times[:n]
    l = left_step_times[:n]
    per_stride = (2 * (r - l) / (r + l)) * 100.0
    if signed:
        return float(per_stride.mean()), per_stride
    else:
        return float(np.abs(per_stride).mean()), per_stride


# ===========================================================================
# Physiologic-plausibility stride filter
# ===========================================================================

def filter_implausible_strides(heel_strike_times: np.ndarray,
                               min_stride_s: float = 0.3,
                               max_stride_s: float = 3.0,
                               ) -> Tuple[np.ndarray, int, int]:
    """Remove heel strikes producing strides outside the plausibility range.

    Intervals shorter than min_stride_s are almost certainly false positives
    (no human walks that fast). Intervals longer than max_stride_s suggest a
    missed detection (trial segment where one heel strike was dropped).

    For too-short intervals we drop the LATER heel strike — it's likely the
    duplicate or artifact. For too-long intervals we currently leave the data
    alone (we don't fabricate missing events) but return a count so the caller
    can warn.

    Returns
    -------
    filtered_times : ndarray        heel strikes with implausibly-short
                                    intervals removed
    n_too_fast     : int            count of intervals < min_stride_s
    n_too_slow     : int            count of intervals > max_stride_s
    """
    if len(heel_strike_times) < 2:
        return heel_strike_times, 0, 0

    times = np.asarray(heel_strike_times)
    intervals = np.diff(times)

    n_too_fast = int(np.sum(intervals < min_stride_s))
    n_too_slow = int(np.sum(intervals > max_stride_s))

    # Drop the LATER peak of each implausibly-short interval
    keep = np.ones(len(times), dtype=bool)
    for i in range(len(intervals)):
        if intervals[i] < min_stride_s:
            keep[i + 1] = False

    return times[keep], n_too_fast, n_too_slow


__all__ = [
    "trim_peaks",
    "compute_step_times",
    "compute_symmetry_index",
    "filter_implausible_strides",
]
