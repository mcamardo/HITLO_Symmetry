"""
hitlo.ankle_angle — sagittal ankle angle from a foot + shank IMU pair.

Ankle angle is the difference between two segment orientations:

    ankle = shank pitch - foot pitch          (zeroed at quiet standing)

The work is getting each segment's pitch reliably from one IMU. Two estimates,
each wrong in a different way, blended:

    gyroscope      integrates to angle, but drifts without bound. Measured
                   foot-sensor bias of -8.8 deg/s integrates to 8.8 degrees of
                   error every second, so the bias MUST be removed.
    accelerometer  gives absolute tilt from gravity with no drift at all, but
                   is only valid when the sensor is not accelerating, which
                   during walking is mostly false.

A complementary filter takes the fast changes from the gyro and lets the
accelerometer pin down the slow drift, but only while the accelerometer is
actually measuring gravity -- see the gate in sagittal_pitch().

NOT WIRED INTO THE COST FUNCTION. hitlo.cost uses step times only. This exists
for offline analysis of recordings that carry a foot sensor.

WHAT IS AND IS NOT VALIDATED
----------------------------
Shape and landmark timing match published gait kinematics: the curve
dorsiflexes through stance, plantarflexes sharply at push-off, and foot-flat
falls where it should in the cycle. Range of motion comes out ~17-20 degrees
against a literature 25-30, most likely mounting compliance -- a sensor taped
to a shoe deforms at push-off in a way one strapped to bone does not.

So: trust the shape and the timing. Do not quote the magnitudes. There is no
goniometer or motion-capture reference behind any of this.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from hitlo.io import SensorStream

# Which gyro axis is sagittal, and which two accelerometer axes span that
# plane, per segment. Derived from measured variance rather than assumed:
# see sagittal_axis().
_INPLANE = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


@dataclass
class AnkleAngleConfig:
    """Everything tunable, in one place."""

    # Complementary filter time constant (s). Above ~0.7 the result is
    # insensitive to this: sweeping 0.2 -> 4.0 s moves range of motion by
    # under a degree once past 0.7, so ROM is not a filter artifact.
    tau: float = 0.7

    # How far |accel| may stray from 1 g before the accelerometer is distrusted
    # entirely. At the gate's edge the filter is pure gyro integration; at the
    # centre it pulls hard toward gravity tilt. 0.30 g leaves the gate open
    # roughly half of walking, which is enough to bound drift.
    gate_width_g: float = 0.30

    # A calibration window must be at least this long and this quiet to count
    # as "standing".
    calib_min_s: float = 5.0
    calib_quiet_dps: float = 25.0

    # Foot-flat detection, used to zero the angle when no standing pose exists.
    footflat_dps: float = 25.0


def sagittal_axis(gyro: np.ndarray, window: Optional[np.ndarray] = None) -> int:
    """Index of the gyro axis carrying sagittal rotation.

    Measured, not assumed: during walking the sagittal axis carries far more
    variance than the other two (typically 130 deg/s against 20-50). Mounting
    orientation differs between sensors and between sessions, so hard-coding
    an axis silently produces a plausible but wrong angle.
    """
    g = np.asarray(gyro, dtype=np.float64)
    if window is not None:
        g = g[window]
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError(f"gyro must be (N, 3), got {np.shape(gyro)}")
    return int(np.argmax(g.std(axis=0)))


def find_calibration_window(stream: SensorStream,
                            cfg: AnkleAngleConfig = AnkleAngleConfig(),
                            ) -> Optional[Tuple[float, float]]:
    """Longest quiet stretch, as (t0, t1) seconds from the start of the stream.

    Returns None if nothing qualifies. A session with no standing period is
    still analysable -- see ankle_angle() -- but the zero point is then not
    comparable across trials.
    """
    t = np.asarray(stream.timestamps, dtype=np.float64)
    t = t - t[0]
    if len(t) < 2:
        return None
    fs = 1.0 / float(np.median(np.diff(t)))
    mag = np.linalg.norm(np.asarray(stream.gyro, dtype=np.float64), axis=1)
    k = max(int(fs), 1)
    smooth = np.convolve(mag, np.ones(k) / k, mode="same")

    quiet = np.flatnonzero(smooth < cfg.calib_quiet_dps)
    if len(quiet) < cfg.calib_min_s * fs:
        return None
    runs = np.split(quiet, np.flatnonzero(np.diff(quiet) > 3) + 1)
    best = max(runs, key=len)
    if len(best) < cfg.calib_min_s * fs:
        return None
    return float(t[best[0]]), float(t[best[-1]])


def sagittal_pitch(accel: np.ndarray,
                   gyro: np.ndarray,
                   fs: float,
                   ref: np.ndarray,
                   cfg: AnkleAngleConfig = AnkleAngleConfig(),
                   ) -> np.ndarray:
    """One segment's pitch in its sagittal plane, in degrees.

    `ref` is a boolean mask selecting samples used to estimate gyro bias. Over
    a quiet stand the mean angular velocity IS the bias. Over whole gait cycles
    it is also the bias, because the limb returns to the same orientation each
    stride and the net rotation per cycle is zero -- which is what makes a
    recording without a standing pose still usable.
    """
    a = np.asarray(accel, dtype=np.float64)
    g = np.asarray(gyro, dtype=np.float64)
    if a.shape != g.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"accel and gyro must both be (N, 3); got "
                         f"{a.shape} and {g.shape}")
    if not (np.isfinite(a).all() and np.isfinite(g).all()):
        raise ValueError("accel/gyro contain non-finite samples")
    if ref.sum() < 2:
        raise ValueError("reference window selects fewer than 2 samples")

    ax = sagittal_axis(g)
    p, q = _INPLANE[ax]

    w = g[:, ax] - g[ref, ax].mean()                 # bias removed
    tilt = np.degrees(np.arctan2(a[:, p], a[:, q]))  # gravity tilt in-plane

    amag = np.linalg.norm(a, axis=1)
    g0 = float(np.median(amag[ref]))                 # this sensor's 1 g
    gate = np.clip(1.0 - np.abs(amag - g0) / cfg.gate_width_g, 0.0, 1.0)

    dt = 1.0 / float(fs)
    alpha = cfg.tau / (cfg.tau + dt)
    th = np.empty(len(w), dtype=np.float64)
    th[0] = tilt[0]
    for i in range(1, len(w)):
        a_i = 1.0 - (1.0 - alpha) * gate[i]          # gate 0 -> pure gyro
        th[i] = a_i * (th[i - 1] + w[i] * dt) + (1.0 - a_i) * tilt[i]
    return th


def ankle_angle(foot: SensorStream,
                shank: SensorStream,
                cfg: AnkleAngleConfig = AnkleAngleConfig(),
                calib: Optional[Tuple[float, float]] = None,
                ) -> Dict[str, object]:
    """Sagittal ankle angle for one leg. Dorsiflexion positive.

    Parameters
    ----------
    foot, shank : SensorStream
        Same leg, same recording. Both must carry gyroscope data.
    calib : (t0, t1), optional
        Quiet-standing window in seconds from the start. Found automatically
        if omitted.

    Returns
    -------
    dict with:
        angle       degrees, same length as the streams, dorsiflexion positive
        t           seconds from the start of the recording
        calib       the window used, or None
        zero        'standing' or 'foot-flat' -- how the zero was set
        note        what that means for interpreting the numbers
    """
    for nm, s in (("foot", foot), ("shank", shank)):
        if not getattr(s, "has_gyro", False):
            raise ValueError(
                f"{nm} stream '{s.name}' carries no gyroscope. Ankle angle "
                f"needs angular velocity from both segments; an accelerometer "
                f"alone cannot separate tilt from linear acceleration during "
                f"walking.")
    n = len(foot.timestamps)
    if len(shank.timestamps) != n:
        raise ValueError(f"foot and shank differ in length ({n} vs "
                         f"{len(shank.timestamps)}); they must come from the "
                         f"same multiplexed stream")

    t = np.asarray(shank.timestamps, dtype=np.float64)
    t = t - t[0]
    fs = 1.0 / float(np.median(np.diff(t)))

    if calib is None:
        calib = find_calibration_window(shank, cfg)

    if calib is not None:
        ref = (t >= calib[0]) & (t <= calib[1])
        zero, note = "standing", (
            "Zeroed at a quiet standing pose, so 0 degrees means this "
            "subject's standing posture. Comparable across trials that share "
            "a standing period.")
    else:
        # No standing pose. Bias still recoverable from whole gait cycles, but
        # the zero point is not: fall back to the mean over foot-flat, which is
        # a defensible neutral but NOT the same reference as standing.
        gmag = np.linalg.norm(np.asarray(foot.gyro, dtype=np.float64), axis=1)
        ref = gmag < cfg.footflat_dps
        if ref.sum() < 2:
            raise ValueError("no standing window and no foot-flat samples; "
                             "cannot establish a reference")
        zero, note = "foot-flat", (
            "No quiet standing period in this recording. Gyro bias was taken "
            "over foot-flat samples and the zero set to the mean foot-flat "
            "angle. Shape and range of motion are usable; the absolute zero "
            "is NOT comparable with standing-calibrated trials.")

    th_foot = sagittal_pitch(foot.accel, foot.gyro, fs, ref, cfg)
    th_shank = sagittal_pitch(shank.accel, shank.gyro, fs, ref, cfg)

    angle = th_shank - th_foot
    angle = angle - angle[ref].mean()

    return dict(angle=angle, t=t, fs=fs, calib=calib, zero=zero, note=note)


def verify_foot_side(foot: SensorStream,
                     shank_same: SensorStream,
                     shank_other: SensorStream,
                     ) -> Dict[str, object]:
    """Is the foot sensor really on the leg its label claims?

    A foot and the shank ABOVE IT swing together: their angular-velocity
    envelopes peak at the same instant. The contralateral shank peaks half a
    stride away. So cross-correlating the envelopes says which leg the foot is
    actually on, independently of what the channel labels assert.

    This is worth checking every time. A mislabelled foot sensor pairs a foot
    with a shank from the OTHER leg, and the resulting "ankle angle" is the
    difference between two segments that never move together -- it comes out
    large, smooth, and completely wrong. It happened in this project: a sensor
    labelled left_foot was physically on the right foot for a whole session.

    Returns {'agrees': bool, 'lag_same_s': float, 'lag_other_s': float}.
    """
    def env(s):
        v = np.linalg.norm(np.asarray(s.gyro, dtype=np.float64), axis=1)
        return v - v.mean()

    a = env(foot)
    n = len(a)
    t = np.asarray(foot.timestamps, dtype=np.float64)
    fs = 1.0 / float(np.median(np.diff(t)))

    def lag(b):
        c = np.correlate(a, b, "full")
        L = np.arange(-n + 1, n)
        keep = np.abs(L) < int(1.2 * fs)
        return float(L[keep][np.argmax(c[keep])]) / fs

    l_same, l_other = lag(env(shank_same)), lag(env(shank_other))
    return dict(agrees=abs(l_same) < abs(l_other),
                lag_same_s=l_same, lag_other_s=l_other)


def stride_profile(angle: np.ndarray,
                   t: np.ndarray,
                   heel_strikes: np.ndarray,
                   n_points: int = 101,
                   stride_range: Tuple[float, float] = (1.0, 1.9),
                   ) -> Optional[Dict[str, np.ndarray]]:
    """Stride-normalised mean and SD, 0-100% of the gait cycle.

    `heel_strikes` are seconds on the same clock as `t`. Strides outside
    stride_range are dropped as detection failures rather than gait.
    """
    hs = np.asarray(heel_strikes, dtype=np.float64)
    out = []
    for a, b in zip(hs[:-1], hs[1:]):
        if not (stride_range[0] < b - a < stride_range[1]):
            continue
        m = (t >= a) & (t < b)
        if m.sum() < 40:
            continue
        out.append(np.interp(np.linspace(0, 1, n_points),
                             np.linspace(0, 1, int(m.sum())), angle[m]))
    if len(out) < 3:
        return None
    S = np.asarray(out)
    return dict(mean=S.mean(axis=0), sd=S.std(axis=0), strides=S, n=len(S))


__all__ = ["AnkleAngleConfig", "sagittal_axis", "find_calibration_window",
           "sagittal_pitch", "ankle_angle", "verify_foot_side",
           "stride_profile"]
