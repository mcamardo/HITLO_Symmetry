"""
hitlo.symmetry — step-time symmetry index computation.

Given two lists of heel-strike timestamps (one per foot) on a common time
base, interleave them to compute step times, then reduce to the gait
symmetry index (SI).

Step time definitions (standard in gait rehab literature):
    Right step = LEFT heel strike  → next RIGHT heel strike
    Left step  = RIGHT heel strike → next LEFT heel strike

Stride time = one step + the subsequent step (one full gait cycle).

Sign convention: SI is signed by default. The symmetry index preserves
direction information so that downstream optimization can either:
  - Drive SI toward 0 (Aim 2 stroke, "minimize asymmetry"), or
  - Drive SI toward a nonzero target (Aim 1 healthy, "induce target
    asymmetry" via passive band perturbation, following the Patton 2004
    paradigm)

The target value (si_target) is NOT used in this module. This module
returns raw signed SI. The target-distance transformation |SI - si_target|
is applied downstream in hitlo.hil_exo._mean_normalize_y when building
GP inputs. This separation keeps logging readable (you see actual SI
values in logs, not abstract distances from target).

Sign meaning:
    SI > 0  → right step time > left step time  (left leg support-dominant)
    SI < 0  → left step time > right step time  (right leg support-dominant)

For a left-side LegExoNET device imitating left-paretic stroke gait, the
expected perturbation direction is SI < 0 (longer left step time, mirroring
the characteristic short-paretic-step pattern). HILBO targeting is signed
accordingly. For Aim 1 the target is NOT a fixed constant: it is derived
per subject from their own baseline, amplified in whichever direction they
already lean (see compute_baseline_target in apps/hitlo_console.py). For
Aim 2 the target is 0.
"""

from typing import Tuple, List
import numpy as np


# ===========================================================================
# Steady-state trim
# ===========================================================================

def walking_window(left, right, thresh_dps: float = 90.0,
                  min_s: float = 10.0):
    """Longest continuous stretch of walking, as (t_start, t_end) LSL seconds.

    The symmetry index must come from walking. A trial is not walking end to
    end: a subject stands while the operator sets the device, walks, and stops
    while LabRecorder is still running. Trimming a fixed few seconds off each
    end does not remove a 30 s stand in the middle of a recording, and every
    non-walking second contributes either nothing or a spurious event.

    Uses the SHANK streams only, never the feet. Feet swing far harder than
    shanks, so averaging them in raises the activity magnitude and the same
    fixed threshold then opens a wider window -- on a four-sensor recording
    that swallowed the ramp-up and moved SI by 2 points.

    Falls back to the whole recording (returns None) when there is no gyro, or
    nothing clears the threshold for min_s, so the caller keeps its previous
    behaviour rather than losing the trial.
    """
    if not (getattr(left, 'has_gyro', False) and getattr(right, 'has_gyro', False)):
        return None
    t = np.asarray(left.timestamps, dtype=np.float64)
    if len(t) < 2:
        return None
    fs = 1.0 / float(np.median(np.diff(t)))
    act = (np.linalg.norm(np.asarray(left.gyro, dtype=np.float64), axis=1) +
           np.linalg.norm(np.asarray(right.gyro, dtype=np.float64), axis=1)) / 2.0
    k = max(int(fs), 1)
    act = np.convolve(act, np.ones(k) / k, mode='same')
    idx = np.flatnonzero(act > thresh_dps)
    if len(idx) < min_s * fs:
        return None
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > k) + 1)
    best = max(runs, key=len)
    if len(best) < min_s * fs:
        return None
    return float(t[best[0]]), float(t[best[-1]])


def trim_peaks(peak_times: np.ndarray,
               trial_start: float,
               trial_end: float,
               trim_s: float) -> np.ndarray:
    """Drop heel strikes within `trim_s` seconds of the trial start or end.

    Ramp-up and ramp-down strides have systematically different mechanics
    (weaker shank accelerations during startup, transient adaptation when
    a perturbation is first applied) and contaminate the steady-state
    symmetry estimate. Standard practice is to trim 3-5 seconds.

    For the perturbation paradigm (Aim 1 healthy with passive band), the
    first ~10-15 strides also reflect feedback-driven correction rather
    than the adapted steady state, but those are handled at the protocol
    level (longer trial duration) rather than via additional trimming here.
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
    right_step_times : ndarray  gaps L→R (time from left HS to next right HS)
    left_step_times  : ndarray  gaps R→L (time from right HS to next left HS)
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
        SI > 0  → right step time > left step time  (left leg support-dominant)
        SI < 0  → left step time > right step time  (right leg support-dominant)

    Per-stride values are ALWAYS signed (sign information preserved for
    downstream paradigm flexibility). The `signed` flag only controls the
    aggregated return value:

        signed=True  → mean(signed SI per stride)
                       Preserves direction. Use when HILBO is targeting
                       a specific signed asymmetry (e.g., si_target = -10).

        signed=False → mean(|signed SI per stride|)
                       Magnitude only. Use when HILBO is minimizing
                       asymmetry magnitude regardless of direction (legacy
                       behavior; equivalent to si_target = 0 in current
                       hitlo.hil_exo, though using signed=True with
                       si_target=0 is preferred for the unified paradigm).

    For the Patton-paradigm Aim 1 (induce target asymmetry via passive band)
    and Aim 2 (drive stroke gait toward symmetry), use signed=True with the
    appropriate si_target in the BO cost calculation downstream
    (hitlo.hil_exo._mean_normalize_y).

    Parameters
    ----------
    right_step_times, left_step_times : ndarray (seconds)
    signed : bool
        See above. Default True.

    Returns
    -------
    mean_si    : float       aggregated SI across strides (signed or |·|)
    per_stride : ndarray     SI per stride (ALWAYS signed, regardless of flag)
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

    Note: for stroke participants, slower self-selected walking speeds may
    occasionally produce stride times approaching max_stride_s (e.g., 2.0-2.5s
    for severe paretic gait). The 3.0s default is generous but may need
    loosening for very slow walkers — empirical adjustment from pilot data.

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
    "walking_window",
    "trim_peaks",
    "compute_step_times",
    "compute_symmetry_index",
    "filter_implausible_strides",
]