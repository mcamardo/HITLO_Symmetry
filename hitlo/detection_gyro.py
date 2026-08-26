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

import warnings

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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

    # How far after a swing peak to look for the crossing, as a FRACTION of
    # the measured stride. Contact follows the peak by roughly a tenth of a
    # cycle at normal cadence, but the interval scales with cadence, so a
    # fixed number cannot serve both.
    #
    # This was 0.50 s fixed, which is fine at a 1.45 s stride and fails at
    # 2.0 s: on a deliberately slowed trial six left swing peaks had no
    # crossing inside the window because contact had not happened yet. The
    # events were simply dropped, alternation fell to 61%, and the leg looked
    # like it was missing strides.
    max_search_frac: float = 0.45
    max_search_s: float = 0.50        # floor, and used before stride is known

    # Likewise the reversal window: how long after the crossing the signal is
    # given to show a genuine reversal.
    reversal_window_frac: float = 0.12

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
    reversal_window_s: float = 0.15   # floor, and used before stride is known

    # Polarity is taken from the larger lobe (mid-swing is the fastest
    # rotation in the cycle). Real shank recordings show lobe ratios of
    # 1.14-1.82, so this is decisive; below this ratio the amplitudes carry
    # no information and the detector falls back to the event-count rule.
    lobe_ratio_min: float = 1.05

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


def estimate_stride_s(w: np.ndarray, fs: float,
                      lo_s: float = 0.7, hi_s: float = 2.5) -> Optional[float]:
    """Stride period from the signal's own autocorrelation.

    Deliberately independent of detection. Deriving the stride from detected
    events instead is circular: if detection is doubling events, the measured
    interval is half the true stride, and spacing derived from it is useless
    exactly when it is needed most.

    Returns None if no periodicity is found in the plausible range.
    """
    x = np.asarray(w, dtype=np.float64)
    x = x - x.mean()
    if len(x) < int(hi_s * fs) * 2:
        return None
    ac = np.correlate(x, x, mode='full')[len(x) - 1:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    lo, hi = int(lo_s * fs), min(int(hi_s * fs), len(ac) - 1)
    if hi <= lo:
        return None
    k = lo + int(np.argmax(ac[lo:hi]))
    # A real gait peak is a clear one; anything weaker is noise structure.
    if ac[k] < 0.2:
        return None
    return float(k / fs)


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
    # Orient so that MID-SWING IS POSITIVE, identified as the larger lobe.
    # Mid-swing is the fastest rotation in the gait cycle; the stance-phase
    # reversal is always smaller.
    #
    # This previously chose the polarity that produced MORE events, tying on
    # regularity. That is wrong twice over. Both orientations of a periodic
    # signal give similarly many, similarly regular events, so the criterion
    # cannot discriminate -- and the count tiebreak is backwards, because the
    # WRONG lobe yields more events (the stance reversal region contributes
    # extra zero crossings). Measured against accelerometer heel strikes on
    # sub-P012/ses-S001, the old rule picked the wrong lobe on 16 of 16 legs
    # across 8 trials, putting every event 468-647 ms early -- about a third
    # of a cycle, near toe-off rather than contact. Orienting by magnitude
    # gives -77 to -113 ms on the same 16 legs (the crossing genuinely
    # precedes the impact peak the accelerometer sees).
    # ------------------------------------------------------------------
    if cfg.swing_sign is not None:
        first = _detect_one_polarity(w, t, cfg)
        oriented = w
    else:
        hi, lo = float(np.max(w)), abs(float(np.min(w)))
        ratio = max(hi, lo) / max(min(hi, lo), 1e-9)
        if ratio >= cfg.lobe_ratio_min:
            oriented = -w if lo > hi else w
            first = _detect_one_polarity(oriented, t, cfg)
        else:
            # Lobes too close to call. Magnitude would be deciding on noise
            # here, and a wrong choice shifts every event by a third of a
            # cycle -- so fall back to running both and keeping whichever
            # yields more events, tying on regularity. That criterion is a
            # poor discriminator on real gait (see above) but it is the only
            # thing left when the amplitudes carry no information.
            pos = _detect_one_polarity(w, t, cfg)
            neg = _detect_one_polarity(-w, t, cfg)
            n_pos, n_neg = len(pos.heel_strike_times), len(neg.heel_strike_times)
            if n_pos == n_neg:
                use_pos = (_regularity(pos.heel_strike_times) <=
                           _regularity(neg.heel_strike_times))
            else:
                use_pos = n_pos > n_neg
            first, oriented = (pos, w) if use_pos else (neg, -w)
            warnings.warn(
                f"gyro polarity is ambiguous: swing and stance lobes differ by "
                f"only {100 * (ratio - 1):.1f}% ({hi:.0f} vs {lo:.0f} deg/s), "
                f"below the {cfg.lobe_ratio_min:.2f} needed to orient by "
                f"magnitude. Fell back to the event-count rule, which picked "
                f"the wrong lobe on all 16 real legs tested. Set swing_sign "
                f"explicitly if the timing looks wrong.",
                RuntimeWarning, stacklevel=2)

    # Second pass at the subject's own cadence. The first pass only needs to
    # be good enough to estimate the stride; spacing derived from it then
    # rejects the extra peaks that a fixed floor lets through.
    if cfg.adaptive_dist_frac and len(first.heel_strike_times) >= 6:
        from dataclasses import replace
        # Autocorrelation, NOT the detected intervals. If the first pass is
        # doubling events its median interval is half the true stride, so
        # spacing derived from it would fail precisely when it is needed.
        stride = estimate_stride_s(oriented, cfg.fs)
        if stride is None:
            stride = float(np.median(np.diff(np.sort(first.heel_strike_times))))
        if stride and np.isfinite(stride) and stride > 0:
            spacing = max(cfg.adaptive_dist_frac * stride, cfg.min_peak_dist_s)
            search = max(cfg.max_search_frac * stride, cfg.max_search_s)
            revwin = max(cfg.reversal_window_frac * stride, cfg.reversal_window_s)
            second = _detect_one_polarity(
                oriented, t, replace(cfg, min_peak_dist_s=spacing,
                                     max_search_s=search,
                                     reversal_window_s=revwin))
            # Keep it only if it is at least as regular; a subject who really
            # changes cadence mid-trial should not be forced onto one stride.
            if (len(second.heel_strike_times) >= 6 and
                    _regularity(second.heel_strike_times) <=
                    _regularity(first.heel_strike_times)):
                return second
    return first


# ===========================================================================
# Toe-off and gait phases
#
# NOT WIRED INTO THE COST FUNCTION. hitlo.cost uses step times only; nothing
# here reaches the optimizer. This exists so stance- and swing-time symmetry
# can be compared against step-time symmetry offline, on recordings already
# collected, before any decision to change what BO optimizes.
# ===========================================================================

def detect_toe_off_gyro(w: np.ndarray,
                        heel_strike_indices: np.ndarray,
                        cfg: GyroDetectionConfig = GyroDetectionConfig(),
                        ) -> np.ndarray:
    """Toe-off index within each stride, from oriented sagittal velocity.

    Toe-off is the local MINIMUM immediately preceding the swing peak. After
    contact the shank rotates backward (deep negative), then forward slowly
    through mid-stance, dipping once more as the trailing limb unloads before
    accelerating into swing.

    Chosen empirically over two alternatives, both of which were unusable.
    Across six leg-conditions from sub-P012, implied stance percentage:

        max acceleration into swing     30.2 - 86.9%   unstable
        threshold above stance plateau  30.2 - 83.4%   unstable
        local min before swing peak     49.2 - 62.8%   stable, and centred
                                                       near the physiological
                                                       ~60% stance

    DOES NOT CURRENTLY WORK. Do not use these numbers.

    The feature is stable on an AVERAGED gait cycle but not per stride: the
    minimum lands anywhere along the flat mid-stance plateau, and averaging
    hid that variance. Measured on sub-P012 normal walking, which is close to
    symmetric by every other measure, this returns stance of 34.8% of the
    cycle on the left against 52.8% on the right, and a stance symmetry index
    of +36.8%. Stance below 50% is not walking at all -- it means more time
    airborne than grounded.

    Kept because the surrounding plumbing is right and the feature choice is
    the standard one; what is missing is a way to pin toe-off per stride.
    That most likely needs either a foot-mounted sensor or a footswitch for
    ground truth, neither of which the current setup has. gait_phases() warns
    when the result is physiologically impossible rather than returning it
    quietly.

    Returns an array the same length as heel_strike_indices; entries are -1
    where no toe-off could be located in that stride.
    """
    w = np.asarray(w, dtype=np.float64)
    hs = np.asarray(heel_strike_indices, dtype=int)
    out = np.full(len(hs), -1, dtype=int)
    if len(hs) < 2:
        return out

    for j, (i0, i1) in enumerate(zip(hs[:-1], hs[1:])):
        n = i1 - i0
        if n < int(0.5 * cfg.fs):
            continue
        seg = w[i0:i1]
        # Swing peak inside this stride, then the last minimum before it.
        sp = int(np.argmax(seg))
        lo = int(0.30 * n)          # past the early-stance trough
        if sp <= lo:
            continue
        out[j] = i0 + lo + int(np.argmin(seg[lo:sp]))
    return out


def gait_phases(w: np.ndarray,
                heel_strike_indices: np.ndarray,
                time_stamps: np.ndarray,
                cfg: GyroDetectionConfig = GyroDetectionConfig(),
                ) -> Dict[str, np.ndarray]:
    """Stance and swing duration per stride, in seconds.

    stance = heel strike -> toe-off, swing = toe-off -> next heel strike.
    Strides where toe-off could not be located are dropped.
    """
    t = np.asarray(time_stamps, dtype=np.float64)
    hs = np.asarray(heel_strike_indices, dtype=int)
    to = detect_toe_off_gyro(w, hs, cfg)
    stance, swing, stride, frac = [], [], [], []
    for j in range(len(hs) - 1):
        if to[j] < 0:
            continue
        st = float(t[to[j]] - t[hs[j]])
        sw = float(t[hs[j + 1]] - t[to[j]])
        if st <= 0 or sw <= 0:
            continue
        stance.append(st); swing.append(sw)
        stride.append(st + sw); frac.append(100.0 * st / (st + sw))
    frac_a = np.asarray(frac)
    # Walking stance is 55-70% of the cycle. Anything outside that means the
    # toe-off feature was not found, whatever the numbers look like.
    if len(frac_a) and not (50.0 <= float(np.median(frac_a)) <= 75.0):
        import warnings as _w
        _w.warn(
            f"toe-off detection produced a median stance of "
            f"{float(np.median(frac_a)):.1f}% of the gait cycle. Walking is "
            f"55-70%; below 50% would mean more time airborne than grounded. "
            f"These phase durations are not usable -- see detect_toe_off_gyro.",
            RuntimeWarning, stacklevel=2)

    return {
        'stance_s': np.asarray(stance),
        'swing_s': np.asarray(swing),
        'stride_s': np.asarray(stride),
        'stance_pct': frac_a,
        'toe_off_indices': to,
    }


__all__ = [
    "GyroDetectionConfig",
    "sagittal_velocity",
    "estimate_stride_s",
    "detect_heelstrikes_gyro",
    "detect_toe_off_gyro",
    "gait_phases",
]
