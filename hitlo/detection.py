"""
hitlo.detection — heel-strike detection pipeline for shank-mounted IMUs.

This module is the single source of truth for heel-strike detection logic.
Both the live BO cost function (hitlo.cost) and the standalone diagnostic
tool (apps/diagnose_trial.py) use these functions so they stay in sync.

PIPELINE
--------
1. compute_magnitude()       : raw |a| = sqrt(x² + y² + z²), orientation invariant
2. compute_jerk_z()          : lowpass-filter |a|, differentiate, then z-score
                               the result; heel strikes show up as big positive
                               jerk spikes
3. detect_peak_candidates()  : strict threshold + gap-fill recovery on jerk z
4. cluster_keep_last()       : group candidates into gait-cycle clusters; pick
                               the LAST peak per cluster that is (a) above the
                               gravity baseline and (b) followed by a stance
                               region (signal flat near baseline afterwards)
5. detect_heelstrikes_full() : one-call wrapper that runs all of the above

PHYSIOLOGIC JUSTIFICATION
-------------------------
Heel strike is a mechanical impact event: the shank decelerates abruptly and
the accelerometer registers a large magnitude spike above gravity (baseline).
Immediately after heel strike, the foot is planted and the shank is
quasi-stationary, so the accelerometer reads essentially gravity alone —
the signal flattens near 1g. Our detector identifies heel strike as the
peak in each cluster that (a) is a genuine impact (above baseline) and
(b) is followed by this stance signature.

NOTE ON FILTER ORDERING
-----------------------
We filter |a| BEFORE differentiating (textbook ordering). Differentiation
amplifies high-frequency noise, so removing the noise first yields a
cleaner derivative.

The cutoff (45 Hz) sits well above the spectral content of heel-strike
impacts (~5–30 Hz), so the lowpass acts as light noise cleanup rather
than active signal shaping. At cutoffs near the impact band (e.g. 15 Hz),
filtering distorts the impact peak and degrades detection — empirically
validated on P048 run-007: 15 Hz filter-then-diff dropped real heel
strikes and doubled per-stride SI variance, while 45 Hz preserves impact
fidelity (raw |a| and filtered |a| overlap through the impact peak).

NOTE ON CAUSALITY
-----------------
filtfilt runs the filter forward + backward → zero phase delay (no time
shift on detected peaks). This pipeline is OFFLINE — filtfilt uses future
samples and cannot run in real time. For a real-time deployment, substitute
a causal FIR with documented group delay.

References
----------
Voisard et al. 2024, J NeuroEng Rehabil 21:104 (jerk + template method)
Prasanth et al. 2021, Sensors (review of rule-based IMU gait detection)
Trojaniello et al. 2014, J NeuroEng Rehabil 11:152 (shank IMU detection)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# ===========================================================================
# Default detection parameters — validated on P048 runs 001-007
# ===========================================================================

@dataclass
class DetectionConfig:
    """All tunable parameters for the detection pipeline in one place.

    Defaults are validated on healthy asymmetric gait (P048). For stroke
    participants, consider loosening stance_tolerance_pct (stroke gait is
    noisier during stance due to compensatory strategies).
    """
    # Sample rate (Polar H10)
    fs: int = 200

    # Jerk signal preparation — 45 Hz keeps the impact band (5–30 Hz) intact
    # while removing high-frequency sensor noise.
    smooth_cutoff_hz: float = 45.0

    # Peak detection thresholds (in units of jerk z-score standard deviations)
    strict_thresh: float = 0.7      # primary pass: catches most events
    recovery_thresh: float = 1.8    # fallback pass: fills anomalous gaps
    gap_multiplier: float = 1.7     # a gap > this × median is "anomalously long"
    min_peak_dist_s: float = 0.10   # minimum separation between candidate peaks

    # Cluster-keep-last
    cluster_gap_s: float = 0.65     # peaks within this gap = same cluster

    # Stance confirmation (post-peak window)
    stance_buffer_s: float = 0.10       # skip this much after peak (impact decay)
    stance_duration_s: float = 0.20     # window length to check
    stance_tolerance_pct: float = 0.15  # mean abs deviation from baseline /
                                        #   baseline must be < this (e.g. 0.15 = 15%)

    # Edge handling
    drop_edge_singletons: bool = True   # drop 1st/last cluster if it's a lone peak


# ===========================================================================
# Stage 1 — signal magnitude
# ===========================================================================

def compute_magnitude(accel_data: np.ndarray) -> np.ndarray:
    """Tri-axial magnitude |a| = sqrt(x² + y² + z²).

    Orientation-invariant: even if the sensor rotates on the shank between
    trials, the magnitude is unchanged. During stance, |a| ≈ 1 g (gravity only);
    during heel strike impact, |a| spikes well above 1 g.

    Parameters
    ----------
    accel_data : ndarray, shape (N, 3)
        Tri-axial accelerometer samples (x, y, z), any units.

    Returns
    -------
    magnitude : ndarray, shape (N,)
    """
    sig = np.asarray(accel_data).T
    return np.sqrt(sig[0] ** 2 + sig[1] ** 2 + sig[2] ** 2)


# ===========================================================================
# Stage 2 — jerk z-score
# ===========================================================================

def compute_jerk_z(accel_data: np.ndarray,
                   cfg: DetectionConfig = DetectionConfig()
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute smoothed, z-scored jerk signal.

    Pipeline order (filter-first, textbook):
        |a| → lowpass-filter → differentiate → z-score

    Differentiation amplifies high-frequency noise, so removing the noise
    BEFORE differentiation gives a cleaner derivative. Heel strikes turn
    into large positive spikes in the jerk signal because the shank
    decelerates abruptly at impact; toe-off and slow gravity-reorientation
    produce smaller jerk responses.

    The lowpass is a 4th-order Butterworth at 45 Hz (default), applied with
    filtfilt → zero phase delay so detected peaks aren't time-shifted. The
    cutoff sits well above heel-strike impact content (5–30 Hz), so the
    filter does light noise cleanup rather than reshaping the impact.

    The result is z-scored so thresholds can be expressed in units of
    standard deviations above the noise floor (subject-independent).

    NOTE: filtfilt is non-causal (uses future samples). This pipeline is
    offline; real-time deployment would require swapping in a causal FIR
    with documented group delay.

    Parameters
    ----------
    accel_data : ndarray, shape (N, 3)
    cfg        : DetectionConfig

    Returns
    -------
    jerk_z    : ndarray, shape (N,)     z-scored jerk
    magnitude : ndarray, shape (N,)     RAW |a|, returned for downstream use
                                        (cluster-keep-last needs the unfiltered
                                        magnitude for baseline + stance check)
    """
    magnitude = compute_magnitude(accel_data)

    # Filter |a| FIRST, then differentiate.
    # Butterworth lowpass via filtfilt → zero phase delay (no time shift).
    b, a = butter(4, cfg.smooth_cutoff_hz / (0.5 * cfg.fs), btype='low')
    magnitude_sm = filtfilt(b, a, magnitude)

    jerk_sm = np.abs(np.diff(magnitude_sm) * cfg.fs)
    jerk_sm = np.concatenate([[0.0], jerk_sm])  # prepend zero to preserve length

    std = float(np.std(jerk_sm))
    if std < 1e-6:
        # Pathological: flat signal (disconnected sensor?)
        return np.zeros_like(jerk_sm), magnitude

    jerk_z = (jerk_sm - np.mean(jerk_sm)) / std
    return jerk_z, magnitude


# ===========================================================================
# Stage 3 — peak candidates (strict + gap-fill recovery)
# ===========================================================================

def _recover_missed_peaks(jerk_z: np.ndarray,
                          initial_peaks: np.ndarray,
                          cfg: DetectionConfig,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Look for missed peaks in anomalously long gaps.

    If gait is rhythmic and we see a gap > GAP_MULTIPLIER × median, there
    probably was a real heel strike there whose jerk happened to fall below
    the strict threshold. We search that gap with a lower threshold.

    This is preferable to interpolating a "fake" peak because we require
    actual signal evidence — if no peak ≥ recovery_thresh exists in the gap,
    we leave it alone.

    Returns
    -------
    all_peaks : sorted union of strict and recovered peaks
    recovered : just the newly added peaks (for diagnostic plotting)
    """
    if len(initial_peaks) < 3:
        return initial_peaks, np.array([], dtype=int)

    intervals = np.diff(initial_peaks) / cfg.fs
    median_interval = float(np.median(intervals))
    gap_thresh_s = cfg.gap_multiplier * median_interval
    min_dist = int(cfg.min_peak_dist_s * cfg.fs)

    recovered = []
    for i in range(len(initial_peaks) - 1):
        gap_s = (initial_peaks[i + 1] - initial_peaks[i]) / cfg.fs
        if gap_s <= gap_thresh_s:
            continue

        search_start = initial_peaks[i] + min_dist
        search_end = initial_peaks[i + 1] - min_dist
        if search_end <= search_start:
            continue

        segment = jerk_z[search_start:search_end]
        cands, _ = find_peaks(segment, height=cfg.recovery_thresh, distance=min_dist)
        if len(cands) == 0:
            continue

        best_local = cands[np.argmax(segment[cands])]
        recovered.append(search_start + best_local)

    recovered_arr = np.array(recovered, dtype=int)
    all_peaks = np.sort(np.concatenate([initial_peaks, recovered_arr]))
    return all_peaks, recovered_arr


def detect_peak_candidates(jerk_z: np.ndarray,
                           cfg: DetectionConfig = DetectionConfig()
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-pass peak detection on jerk z-score.

    Pass 1 (strict): find_peaks at cfg.strict_thresh. Catches most events.
    Pass 2 (recovery): look in anomalously long gaps at cfg.recovery_thresh.
    Fills in heel strikes that happened to fall below the strict threshold.

    Returns
    -------
    all_candidates : ndarray[int]    merged & sorted peak indices
    strict_peaks   : ndarray[int]    pass-1 peaks (for diagnostics)
    recovered      : ndarray[int]    pass-2 additions (for diagnostics)
    """
    min_dist = int(cfg.min_peak_dist_s * cfg.fs)
    strict_peaks, _ = find_peaks(jerk_z,
                                  height=cfg.strict_thresh,
                                  distance=min_dist)
    all_candidates, recovered = _recover_missed_peaks(jerk_z, strict_peaks, cfg)
    return all_candidates, strict_peaks, recovered


# ===========================================================================
# Stage 4 — cluster-keep-last with stance confirmation
# ===========================================================================

def cluster_keep_last(candidates: np.ndarray,
                      magnitude: np.ndarray,
                      cfg: DetectionConfig = DetectionConfig(),
                      ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Group candidates into gait-cycle clusters, pick one heel strike each.

    For each cluster (peaks within cfg.cluster_gap_s of a neighbor), scan
    from the LAST peak backwards and pick the first one that satisfies:

      (a) Raw magnitude above baseline (not a trough-landing false peak).
          Baseline = median(|a|) across the trial, which approximates 1 g.
          Peaks below baseline are jerk-edge artifacts from near-free-fall
          moments during swing, not real impacts.

      (b) Followed by a stance region. Compute the mean absolute deviation
          from baseline over a post-peak window [buffer, buffer + duration].
          If that MAD < tolerance × baseline, the shank is quasi-stationary
          — confirming heel strike immediately preceded stance.

    "One cluster = one heel strike." The other peaks in the cluster (toe-off,
    pre-impact wobbles) are rejected.

    If no peak in a cluster qualifies, the whole cluster is rejected —
    we never fabricate a heel strike.

    Edge handling: the very first and last cluster, if they are singletons
    (lone peaks), are dropped as likely partial-cycle trial-edge artifacts.

    Parameters
    ----------
    candidates : ndarray[int]    peak indices from detect_peak_candidates
    magnitude  : ndarray         raw |a| for the same trial
    cfg        : DetectionConfig

    Returns
    -------
    accepted     : ndarray[int]             one heel strike index per cluster
    rejected     : ndarray[int]             all other candidate indices
    cluster_info : list of (start, end)     (first, last) candidate per cluster,
                                            useful for diagnostic shading
    """
    if len(candidates) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), []

    candidates = np.asarray(sorted(candidates))
    gap_samples = int(cfg.cluster_gap_s * cfg.fs)
    baseline = float(np.median(magnitude))

    # Precompute stance check as a closure so we can reuse it cleanly
    buf_n = int(cfg.stance_buffer_s * cfg.fs)
    win_n = int(cfg.stance_duration_s * cfg.fs)
    tol = baseline * cfg.stance_tolerance_pct

    def has_stance_after(idx: int) -> bool:
        start = int(idx) + buf_n
        end = start + win_n
        if end >= len(magnitude):
            return True  # too close to recording end, give benefit of doubt
        window = magnitude[start:end]
        mad = float(np.mean(np.abs(window - baseline)))
        return mad <= tol

    accepted_per_cluster: List[int] = []
    cluster_is_singleton: List[bool] = []
    rejected: List[int] = []
    cluster_info: List[Tuple[int, int]] = []

    i = 0
    while i < len(candidates):
        cluster_start = i
        while (i + 1 < len(candidates) and
               candidates[i + 1] - candidates[i] <= gap_samples):
            i += 1
        cluster_end = i

        cluster_members = list(candidates[cluster_start:cluster_end + 1])

        # Scan from LAST peak backwards; first one above baseline AND
        # followed by stance is the heel strike.
        chosen = None
        for idx in reversed(cluster_members):
            idx_i = int(idx)
            if idx_i >= len(magnitude):
                continue
            if magnitude[idx_i] < baseline:
                continue  # trough peak
            if not has_stance_after(idx_i):
                continue  # no stance follows
            chosen = idx_i
            break

        if chosen is None:
            # No valid heel strike in this cluster — reject everything
            for idx in cluster_members:
                rejected.append(int(idx))
        else:
            accepted_per_cluster.append(chosen)
            cluster_is_singleton.append(len(cluster_members) == 1)
            cluster_info.append((int(cluster_members[0]),
                                 int(cluster_members[-1])))
            for idx in cluster_members:
                if int(idx) != chosen:
                    rejected.append(int(idx))

        i += 1

    # Drop edge singletons (partial cycles at trial boundaries)
    if cfg.drop_edge_singletons and accepted_per_cluster:
        if cluster_is_singleton[-1]:
            rejected.append(accepted_per_cluster.pop())
            cluster_is_singleton.pop()
            cluster_info.pop()
        if accepted_per_cluster and cluster_is_singleton[0]:
            rejected.append(accepted_per_cluster.pop(0))
            cluster_is_singleton.pop(0)
            cluster_info.pop(0)

    return (np.array(accepted_per_cluster, dtype=int),
            np.array(rejected, dtype=int),
            cluster_info)


# ===========================================================================
# Stage 5 — one-call wrapper
# ===========================================================================

@dataclass
class DetectionResult:
    """All outputs of the detection pipeline for one sensor/trial.

    Split into `accepted` (heel strikes) and various diagnostic arrays so that
    downstream code can use just the heel strikes while diagnostic/QC code
    can show the full pipeline state.
    """
    heel_strike_indices: np.ndarray       # accepted peaks (sample indices)
    heel_strike_times:   np.ndarray       # LSL timestamps at those indices
    all_candidates:      np.ndarray       # pre-clustering peaks
    strict_peaks:        np.ndarray       # pass-1 detections
    recovered_peaks:     np.ndarray       # pass-2 additions
    rejected_peaks:      np.ndarray       # candidates that failed cluster rules
    cluster_info:        List[Tuple[int, int]]   # for diagnostic shading
    jerk_z:              np.ndarray       # z-scored jerk signal
    magnitude:           np.ndarray       # raw |a|


def detect_heelstrikes_full(accel_data: np.ndarray,
                            time_stamps: np.ndarray,
                            cfg: DetectionConfig = DetectionConfig(),
                            ) -> DetectionResult:
    """Run the full detection pipeline for one sensor.

    This is the primary entry point. Both symmetry_cost and the diagnostic
    tool call this so they produce identical heel-strike lists.

    Parameters
    ----------
    accel_data  : ndarray, shape (N, 3)
        Tri-axial accelerometer samples.
    time_stamps : ndarray, shape (N,)
        LSL timestamps for each sample. Heel-strike times come from these.
    cfg         : DetectionConfig

    Returns
    -------
    DetectionResult
    """
    # Stages 1-2
    jerk_z, magnitude = compute_jerk_z(accel_data, cfg=cfg)

    # Stage 3
    candidates, strict_peaks, recovered = detect_peak_candidates(jerk_z, cfg=cfg)

    # Stage 4
    accepted, rejected, cluster_info = cluster_keep_last(candidates, magnitude, cfg=cfg)

    # Map indices → LSL timestamps
    if len(accepted) > 0:
        safe = accepted[accepted < len(time_stamps)]
        heel_strike_times = time_stamps[safe]
    else:
        heel_strike_times = np.array([])

    return DetectionResult(
        heel_strike_indices=accepted,
        heel_strike_times=heel_strike_times,
        all_candidates=candidates,
        strict_peaks=strict_peaks,
        recovered_peaks=recovered,
        rejected_peaks=rejected,
        cluster_info=cluster_info,
        jerk_z=jerk_z,
        magnitude=magnitude,
    )


__all__ = [
    "DetectionConfig",
    "DetectionResult",
    "compute_magnitude",
    "compute_jerk_z",
    "detect_peak_candidates",
    "cluster_keep_last",
    "detect_heelstrikes_full",
]
