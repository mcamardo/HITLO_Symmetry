"""
hitlo.detection_gyro — heel strike from shank angular velocity.

WHY THIS EXISTS ALONGSIDE hitlo.detection
-----------------------------------------
The accelerometer detector infers contact from an impact transient: filter
|a|, differentiate to jerk, threshold, cluster the resulting peaks, and pick
one per cluster. It works, but the event it finds is the impact *shock*, and
which peak represents it is a judgement made among several similar ones. On a
limb where that shock is damped — a compliant mount, cushioned footwear, an
exoskeleton in the load path — the peaks become near-equal and the choice
degrades. Measured on sub-P997/P996: the accepted peak beat its rejected
competitors by 4.4 SD on the free leg and 0.16 SD on the instrumented one.

Sagittal shank angular velocity does not have that ambiguity. The shank
rotates forward rapidly through swing, producing one large unmistakable peak,
then reverses sharply as the foot contacts the ground. Initial contact is the
negative-going ZERO CROSSING immediately after that peak — a sign change, not
an amplitude comparison, so there is nothing to choose between and no
threshold that damping can push the signal below.

Two consequences worth being explicit about:

1. There is no clustering, no stance confirmation and no recovery pass. Those
   exist to disambiguate competing peaks. A zero crossing is unique within a
   cycle, so they have nothing to do.

2. This finds a DIFFERENT INSTANT than the accelerometer detector. The zero
   crossing is initial contact; the jerk peak is the shock that follows it,
   typically some tens of milliseconds later. Symmetry indices from the two
   detectors are therefore not directly comparable, and a baseline collected
   with one must not be carried into a session run with the other.

Reference: the swing-peak-then-zero-crossing rule is the standard gyroscopic
gait-event method (e.g. Aminian 2002; Salarian 2004), widely validated
against force plates and footswitches.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from hitlo.detection import DetectionResult


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class GyroDetectionConfig:
    """Tunables for gyroscope-based heel-strike detection."""

    # Sample rate. As with DetectionConfig this MUST match the hardware;
    # use for_stream() rather than trusting the default.
    fs: int = 148                     # Trigno Avanti default

    # Which gyro column is the sagittal axis (rotation about the medio-lateral
    # axis). Depends on how the sensor is mounted, so it is configurable and
    # the sign is resolved automatically below.
    sagittal_axis: int = 2            # 0=x, 1=y, 2=z

    # Sign convention. Gyro polarity depends on mounting orientation, so by
    # default the direction of swing is inferred from the data instead of
    # assumed. Set explicitly (+1/-1) to pin it.
    swing_sign: Optional[int] = None

    # Angular velocity is far smoother than jerk, so this is gentle noise
    # cleanup rather than shaping. Keep it well above the swing bandwidth
    # (~1-3 Hz) and above the contact reversal (~10 Hz).
    lowpass_hz: float = 15.0

    # Swing-peak selection. The threshold is a FRACTION of the trial's own
    # peak swing rate rather than an absolute deg/s, so it transfers across
    # subjects and walking speeds without retuning.
    swing_peak_frac: float = 0.35
    min_peak_dist_s: float = 0.40     # floor for the first pass only

    # After a first pass establishes the subject's actual stride time, the
    # detector re-runs with the peak spacing set to this fraction of it.
    #
    # A fixed floor cannot do this job: 0.40 s is a sensible lower bound, but
    # at a 1.42 s stride it admits three "swing peaks" per cycle, so any
    # post-contact ringing on a limb becomes a second detected contact. That
    # was observed on real data -- one leg returning 62 events to the other's
    # 56 over the same window, purely from ringing between strides.
    #
    # Set to None to disable the second pass and keep the fixed floor.
    adaptive_dist_frac: Optional[float] = 0.60

    # How far after a swing peak to look for the crossing. Contact follows
    # the peak closely; a wider window risks catching the next cycle.
    max_search_s: float = 0.50

    # Reject a crossing that is not followed by a real reversal — that is
    # drift or a stance wobble rather than foot contact. Expressed as a
    # fraction of the trial's own peak swing rate that the signal must reach
    # in the NEGATIVE direction shortly after crossing.
    #
    # Deliberately an amplitude test, not a slope test: a per-sample slope
    # threshold is not sample-rate independent — the same signal digitized
    # faster has smaller per-sample steps and would fail a fixed slope floor
    # while being identical physically.
    min_reversal_frac: float = 0.08
    reversal_window_s: float = 0.15

    def with_fs(self, fs: float) -> "GyroDetectionConfig":
        from dataclasses import replace
        return replace(self, fs=int(round(float(fs))))

    def for_stream(self, stream, tolerance: float = 0.05,
                   warn: bool = True) -> "GyroDetectionConfig":
        """Config matched to the rate the stream actually delivered.

        Same rationale as DetectionConfig.for_stream: every window here is in
        seconds and converted with fs.
        """
        fs = float(getattr(stream, "actual_fs", self.fs))
        if not np.isfinite(fs) or fs <= 0:
            return self
        if warn and abs(fs - self.fs) / max(self.fs, 1) > tolerance:
            import warnings as _w
            _w.warn(
                f"GyroDetectionConfig.fs is {self.fs} Hz but the stream "
                f"measured {fs:.1f} Hz. Using the measured rate.",
                RuntimeWarning, stacklevel=2)
        return self.with_fs(fs)


# ===========================================================================
# Signal preparation
# ===========================================================================

def sagittal_velocity(gyro: np.ndarray,
                      cfg: GyroDetectionConfig = GyroDetectionConfig(),
                      ) -> np.ndarray:
    """Filtered sagittal angular velocity, oriented so swing is POSITIVE.

    Mounting orientation decides the raw sign, so rather than require the
    operator to get it right, infer it: swing is the largest excursion in the
    gait cycle, so whichever polarity holds the extreme values is swing.
    """
    g = np.asarray(gyro, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] <= cfg.sagittal_axis:
        raise ValueError(
            f"gyro must be (N, 3) with a column {cfg.sagittal_axis}; "
            f"got shape {g.shape}")

    w = g[:, cfg.sagittal_axis]
    if not np.all(np.isfinite(w)):
        n_bad = int((~np.isfinite(w)).sum())
        raise ValueError(
            f"sagittal gyro has {n_bad}/{len(w)} non-finite samples")

    nyq = 0.5 * cfg.fs
    if cfg.lowpass_hz < nyq:
        b, a = butter(4, cfg.lowpass_hz / nyq, btype='low')
        w = filtfilt(b, a, w)

    if cfg.swing_sign is not None:
        return w * float(cfg.swing_sign)
    return w  # polarity resolved by the detector, which can test both


# ===========================================================================
# Detection
# ===========================================================================

def _interpolated_crossing(w: np.ndarray, i: int) -> float:
    """Sub-sample position of the zero crossing between i and i+1.

    The crossing rarely lands on a sample. At 148 Hz one sample is 6.8 ms,
    and since symmetry index moves roughly 0.28 points per millisecond of
    timing error, rounding to the nearest sample would inject about 2 points
    of noise for free. Linear interpolation removes it.
    """
    y0, y1 = float(w[i]), float(w[i + 1])
    if y0 == y1:
        return float(i)
    return float(i) + y0 / (y0 - y1)


def _detect_one_polarity(w: np.ndarray, t: np.ndarray,
                         cfg: GyroDetectionConfig) -> DetectionResult:
    """Run the swing-peak -> zero-crossing rule on an already-oriented signal."""
    n = len(w)
    empty = np.array([], dtype=int)

    peak_ref = float(np.percentile(w, 95))
    height = cfg.swing_peak_frac * peak_ref
    peaks, _ = find_peaks(w, height=height,
                          distance=max(int(cfg.min_peak_dist_s * cfg.fs), 1))

    max_search = int(cfg.max_search_s * cfg.fs)
    rev_floor = cfg.min_reversal_frac * peak_ref
    rev_win = max(int(cfg.reversal_window_s * cfg.fs), 2)

    idx: List[int] = []
    times: List[float] = []
    used: List[int] = []
    unused: List[int] = []
    pairs: List[Tuple[int, int]] = []

    for p in peaks:
        stop = min(int(p) + max_search, n - 1)
        seg = w[int(p):stop + 1]
        below = np.flatnonzero((seg[:-1] >= 0) & (seg[1:] < 0))
        if len(below) == 0:
            unused.append(int(p))
            continue
        k = int(p) + int(below[0])
        tail = w[k + 1:min(k + 1 + rev_win, n)]
        if len(tail) == 0 or float(np.min(tail)) > -rev_floor:
            unused.append(int(p))
            continue
        frac = _interpolated_crossing(w, k)
        lo = int(np.floor(frac))
        hi = min(lo + 1, n - 1)
        alpha = frac - lo
        idx.append(int(round(frac)))
        times.append(float(t[lo] * (1.0 - alpha) + t[hi] * alpha))
        used.append(int(p))
        pairs.append((int(p), k))

    order = np.argsort(times) if times else np.array([], dtype=int)
    idx_a = np.asarray(idx, dtype=int)[order] if len(idx) else empty
    times_a = np.asarray(times, float)[order] if len(times) else np.array([])

    sd = float(np.std(w))
    return DetectionResult(
        heel_strike_indices=idx_a,
        heel_strike_times=times_a,
        all_candidates=np.asarray(peaks, dtype=int),
        strict_peaks=np.asarray(used, dtype=int),
        recovered_peaks=empty,
        rejected_peaks=np.asarray(unused, dtype=int),
        cluster_info=pairs,
        jerk_z=(w - float(np.mean(w))) / (sd + 1e-9),
        magnitude=w,
    )


def _regularity(times: np.ndarray) -> float:
    """How periodic a set of events is. Lower is better; inf if too few.

    Used to choose between polarities. Gait is strongly periodic, so the
    orientation that produces evenly spaced events is the one that found
    swing rather than stance.
    """
    if len(times) < 4:
        return float('inf')
    iv = np.diff(np.sort(times))
    m = float(np.median(iv))
    if m <= 0:
        return float('inf')
    return float(np.median(np.abs(iv - m)) / m)


def detect_heelstrikes_gyro(gyro: np.ndarray,
                            time_stamps: np.ndarray,
                            cfg: GyroDetectionConfig = GyroDetectionConfig(),
                            ) -> DetectionResult:
    """Detect initial contact from shank angular velocity.

    Returns the same DetectionResult as the accelerometer detector so the
    optimizer, cost function and diagnostics do not care which produced it.
    Field meanings are analogous rather than identical:

        heel_strike_indices  sample index of each negative-going crossing
        heel_strike_times    interpolated LSL time of contact
        all_candidates       swing peaks considered
        strict_peaks         swing peaks that yielded a crossing
        recovered_peaks      always empty (no recovery pass is needed)
        rejected_peaks       swing peaks with no valid crossing after them
        cluster_info         (swing peak, crossing) index pairs
        jerk_z               z-scored sagittal velocity, for plotting
        magnitude            filtered sagittal velocity (deg/s)
    """
    w = sagittal_velocity(gyro, cfg)
    t = np.asarray(time_stamps, dtype=np.float64)
    n = len(w)
    empty = np.array([], dtype=int)

    if n < cfg.fs or len(t) != n:
        return DetectionResult(
            heel_strike_indices=empty, heel_strike_times=np.array([]),
            all_candidates=empty, strict_peaks=empty, recovered_peaks=empty,
            rejected_peaks=empty, cluster_info=[],
            jerk_z=np.zeros(n), magnitude=w)

    # ------------------------------------------------------------------
    # Polarity. Gyro sign depends on how the sensor was clipped to the
    # shank, and getting it wrong locks onto the stance reversal instead of
    # swing -- every event lands at the wrong point in the cycle, while
    # still looking like a clean periodic detection.
    #
    # Rather than infer it from which excursion is larger (which fails
    # whenever the two lobes are comparable, and decides on noise when they
    # are equal), run BOTH polarities and keep the one that produces more
    # events, breaking ties on how evenly spaced they are. Gait is strongly
    # periodic, so the correct orientation is self-evident from the result.
    # ------------------------------------------------------------------
    if cfg.swing_sign is not None:
        first = _detect_one_polarity(w, t, cfg)
        oriented = w
    else:
        pos = _detect_one_polarity(w, t, cfg)
        neg = _detect_one_polarity(-w, t, cfg)
        n_pos, n_neg = len(pos.heel_strike_times), len(neg.heel_strike_times)
        if n_pos == n_neg:
            use_pos = (_regularity(pos.heel_strike_times) <=
                       _regularity(neg.heel_strike_times))
        else:
            use_pos = n_pos > n_neg
        first, oriented = (pos, w) if use_pos else (neg, -w)

    # Second pass at the subject's own cadence. The first pass only needs to
    # be good enough to estimate the stride; spacing derived from it then
    # rejects the extra peaks that a fixed floor lets through.
    if cfg.adaptive_dist_frac and len(first.heel_strike_times) >= 6:
        from dataclasses import replace
        stride = float(np.median(np.diff(np.sort(first.heel_strike_times))))
        if np.isfinite(stride) and stride > 0:
            spacing = max(cfg.adaptive_dist_frac * stride, cfg.min_peak_dist_s)
            second = _detect_one_polarity(
                oriented, t, replace(cfg, min_peak_dist_s=spacing))
            # Keep it only if it is at least as regular; a subject who really
            # changes cadence mid-trial should not be forced onto one stride.
            if (len(second.heel_strike_times) >= 6 and
                    _regularity(second.heel_strike_times) <=
                    _regularity(first.heel_strike_times)):
                return second
    return first


__all__ = [
    "GyroDetectionConfig",
    "sagittal_velocity",
    "detect_heelstrikes_gyro",
]
