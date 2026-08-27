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
There are two paths in this module, and they are not equally trustworthy.

The path described above -- derive each segment's axis from its own motion
during walking, then difference two pitches -- gives the right SHAPE and the
right landmark timing, and a range of motion that is simply wrong. It reached
140 degrees on P017 against a literature 25-30. The cause is not mounting
compliance, which is what the earlier version of this note guessed: the two
axes are derived independently and are not the same physical axis, so their
difference is not a joint angle at all. Read that path for timing only.

The second path -- functional_calibration() and ankle_angle_functional(),
at the bottom of this file -- measures the axis instead of inferring it, and
recovers a known excursion to within 0.03 degrees in simulation across four
sensor mountings. It requires a short calibration movement to have been
recorded, and refuses to run without one.

Neither path has been checked against a goniometer or motion capture.
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

    # --- functional calibration (see functional_calibration) ---------------
    # A segment counts as "moving" above func_move_dps and "still" below
    # func_still_dps. The gap between them is deliberate: samples in between
    # are unclassified rather than forced into a bin.
    func_move_dps: float = 30.0
    func_still_dps: float = 15.0
    func_min_s: float = 4.0

    # PC1 must beat PC2 by this factor for a movement to count as isolating
    # one axis. Below it the subject moved in more than one plane and the
    # axis is not identified.
    func_axis_margin: float = 2.5

    # In the rigid-body part the two projections must agree this well, and
    # their amplitude ratio must sit this close to 1.
    func_corr_min: float = 0.90
    func_gain_tol: float = 0.30


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

    # Gravity tilt in the sagittal plane, expressed RELATIVE TO THE
    # CALIBRATION POSE and wrapped to (-180, 180].
    #
    # Taking the raw atan2 breaks whenever the mounting happens to put the
    # tilt near the +/-180 discontinuity: the signal then jumps 360 degrees
    # every time it crosses, the complementary filter chases each jump, and
    # the angle never recovers. That is not hypothetical -- a session where
    # the shank sensors were mounted inverted sat at -145 degrees and crossed
    # the wrap 1087 times in 89 s of walking, producing a 134 degree range of
    # motion. The same code on the same subject the day before, with the
    # sensors the other way up, sat at -1.2 degrees and never wrapped.
    #
    # Referencing to the calibration pose puts the working range around zero
    # for ANY mounting, so the discontinuity is half a turn away from where
    # the signal lives. The zero it establishes is the one we want anyway.
    raw = np.arctan2(a[:, p], a[:, q])
    ref_ang = np.arctan2(np.sin(raw[ref]).mean(), np.cos(raw[ref]).mean())
    tilt = np.degrees(np.angle(np.exp(1j * (raw - ref_ang))))

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


# The pose. Written out because a calibration is only as good as the posture
# it was captured in, and "stand still" is not specific enough -- a subject
# resting weight on one leg, or with a knee soft, gives a neutral that does not
# correspond to the neutral their gait is measured against.
CALIBRATION_POSE = (
    "Stand tall and still, weight even on both feet, feet flat and pointing "
    "forward, knees straight but not locked, arms at your sides. Look ahead, "
    "not down. Hold for 10 seconds."
)


def validate_calibration_pose(foot: SensorStream,
                              shank: SensorStream,
                              window: Tuple[float, float],
                              cfg: AnkleAngleConfig = AnkleAngleConfig(),
                              ) -> Dict[str, object]:
    """Was that actually a still, upright, neutral pose?

    A calibration inherits every error in the posture it was captured in, and
    silently: a subject shifting weight, or standing with one knee soft, still
    produces a clean-looking zero. These checks catch the cases that matter.

        duration    long enough to average out noise
        stillness   the subject was not swaying or shifting
        gravity     each sensor reads ~1 g, so it is upright and working
        shank tilt  the shank is near vertical, as it is in standing

    Returns {'ok': bool, 'checks': [...]}. Each check carries a level of
    'ok' | 'warn' | 'fail' and a human-readable reason.
    """
    t = np.asarray(shank.timestamps, dtype=np.float64)
    t = t - t[0]
    m = (t >= window[0]) & (t <= window[1])
    checks = []

    dur = float(window[1] - window[0])
    checks.append(dict(
        name="duration", level=("ok" if dur >= cfg.calib_min_s else "fail"),
        detail=f"{dur:.1f} s held",
        why=f"needs at least {cfg.calib_min_s:.0f} s to average out noise"))

    for nm, stream in (("foot", foot), ("shank", shank)):
        g = np.asarray(stream.gyro, dtype=np.float64)[m]
        a = np.asarray(stream.accel, dtype=np.float64)[m]
        if len(g) < 2:
            checks.append(dict(name=f"{nm} stillness", level="fail",
                               detail="no samples in the window", why=""))
            continue

        rms = float(np.sqrt((np.linalg.norm(g, axis=1) ** 2).mean()))
        lvl = "ok" if rms < cfg.calib_quiet_dps else (
            "warn" if rms < 2 * cfg.calib_quiet_dps else "fail")
        checks.append(dict(
            name=f"{nm} stillness", level=lvl,
            detail=f"{rms:.1f} deg/s RMS",
            why=("swaying or shifting weight during the hold biases the zero"
                 if lvl != "ok" else "still")))

        gmag = float(np.linalg.norm(a, axis=1).mean())
        lvl = "ok" if 0.90 <= gmag <= 1.10 else (
            "warn" if 0.80 <= gmag <= 1.20 else "fail")
        checks.append(dict(
            name=f"{nm} gravity", level=lvl,
            detail=f"{gmag:.3f} g",
            why=("should read ~1 g when still; a large deviation means the "
                 "sensor is moving, mis-scaled, or faulty"
                 if lvl != "ok" else "reads 1 g")))

    a = np.asarray(shank.accel, dtype=np.float64)[m].mean(axis=0)
    n = a / max(float(np.linalg.norm(a)), 1e-9)
    tilt = float(np.degrees(np.arccos(np.clip(np.max(np.abs(n)), -1, 1))))
    lvl = "ok" if tilt < 15 else ("warn" if tilt < 30 else "fail")
    checks.append(dict(
        name="shank upright", level=lvl, detail=f"{tilt:.1f} deg off axis",
        why=("the shank is near vertical in standing, so a large angle means "
             "the sensor is rotated on the limb or the pose was not upright"
             if lvl != "ok" else "near vertical")))

    return dict(ok=not any(c["level"] == "fail" for c in checks),
                warn=any(c["level"] == "warn" for c in checks),
                checks=checks, pose=CALIBRATION_POSE)


def session_calibration(foot: SensorStream,
                        shank: SensorStream,
                        cfg: AnkleAngleConfig = AnkleAngleConfig(),
                        calib: Optional[Tuple[float, float]] = None,
                        ) -> Dict[str, object]:
    """Capture a calibration once, to reuse for the rest of a session.

    Standing still before every trial is not always practical. Measured on a
    recording with two quiet stands 105 s apart, reusing the earlier one costs
    a mean offset of **0.89 degrees** and changes range of motion by 0.02
    degrees, with the stride-averaged shape correlating at r = 0.9999. Gyro
    bias itself barely moves: 0.09 deg/s on the foot and 0.01 on the shank
    over that interval.

    So one calibration per mounting is fine. What it does NOT survive is the
    sensor moving: re-strap or re-tape anything and the reference is stale,
    because the zero encodes where the sensor sits on the limb.

    Returns a dict to hand to ankle_angle(calibration=...). It records which
    axis and which stream it came from, and ankle_angle refuses to apply it to
    a differently-mounted sensor rather than silently producing a plausible
    wrong answer.
    """
    if calib is None:
        calib = find_calibration_window(shank, cfg)
    if calib is None:
        raise ValueError(
            "no quiet standing period found, so there is nothing to capture. "
            "Record ~10 s of standing at the start of a trial, or pass an "
            "explicit window.")
    res = ankle_angle(foot, shank, cfg, calib=calib)
    t = res["t"]
    ref = (t >= calib[0]) & (t <= calib[1])

    def _cap(stream):
        g = np.asarray(stream.gyro, dtype=np.float64)
        a = np.asarray(stream.accel, dtype=np.float64)
        ax = sagittal_axis(g)
        p_, q_ = _INPLANE[ax]
        # The tilt reference must be stored too, not just bias and gravity.
        # sagittal_pitch measures tilt RELATIVE to the calibration pose (see
        # the atan2 wrap note there), so a reuse path that recomputed a raw
        # atan2 would silently disagree with the direct path -- it did, by 17
        # degrees, until the regression test caught it.
        raw = np.arctan2(a[ref, p_], a[ref, q_])
        return dict(axis=int(ax), bias=float(g[ref, ax].mean()),
                    g0=float(np.median(np.linalg.norm(a[ref], axis=1))),
                    ref_ang=float(np.arctan2(np.sin(raw).mean(),
                                             np.cos(raw).mean())),
                    name=str(stream.name))

    return dict(foot=_cap(foot), shank=_cap(shank),
                zero=float((res["angle"] + res["_zero_offset"])[ref].mean()),
                window=calib)


def ankle_angle(foot: SensorStream,
                shank: SensorStream,
                cfg: AnkleAngleConfig = AnkleAngleConfig(),
                calib: Optional[Tuple[float, float]] = None,
                calibration: Optional[Dict[str, object]] = None,
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

    if calibration is not None:
        # Reuse a calibration captured earlier in the session. Refuse if the
        # sensors are not the ones it came from: applying another sensor's
        # zero produces a smooth, plausible, wrong angle.
        for nm, cap, stream in (("foot", calibration["foot"], foot),
                                ("shank", calibration["shank"], shank)):
            if cap["name"] != stream.name:
                raise ValueError(
                    f"calibration was captured from {nm} stream "
                    f"'{cap['name']}' but this is '{stream.name}'. A zero "
                    f"encodes where a sensor sits on the limb; applying it to "
                    f"a different sensor gives a plausible wrong answer.")
        th_foot = _pitch_with(foot, fs, calibration["foot"], cfg)
        th_shank = _pitch_with(shank, fs, calibration["shank"], cfg)
        angle = (th_shank - th_foot) - float(calibration["zero"])
        return dict(angle=angle, t=t, fs=fs, calib=calibration["window"],
                    zero="session", _zero_offset=float(calibration["zero"]),
                    note=("Reusing a calibration captured earlier this session "
                          f"({calibration['window'][0]:.0f}-"
                          f"{calibration['window'][1]:.0f} s of that recording). "
                          "Measured cost of reuse across ~105 s is a 0.9 degree "
                          "offset with shape unchanged. Recapture if a sensor "
                          "was re-mounted."))

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
    offset = float(angle[ref].mean())
    angle = angle - offset

    return dict(angle=angle, t=t, fs=fs, calib=calib, zero=zero, note=note,
                _zero_offset=offset)


def _pitch_with(stream: SensorStream, fs: float, cap: Dict[str, object],
                cfg: AnkleAngleConfig) -> np.ndarray:
    """sagittal_pitch using a stored bias and gravity magnitude."""
    a = np.asarray(stream.accel, dtype=np.float64)
    g = np.asarray(stream.gyro, dtype=np.float64)
    ax = int(cap["axis"])
    p, q = _INPLANE[ax]
    w = g[:, ax] - float(cap["bias"])
    raw = np.arctan2(a[:, p], a[:, q])
    tilt = np.degrees(np.angle(np.exp(1j * (raw - float(cap["ref_ang"])))))
    amag = np.linalg.norm(a, axis=1)
    gate = np.clip(1.0 - np.abs(amag - float(cap["g0"])) / cfg.gate_width_g, 0.0, 1.0)
    dt = 1.0 / float(fs)
    alpha = cfg.tau / (cfg.tau + dt)
    th = np.empty(len(w), dtype=np.float64)
    th[0] = tilt[0]
    for i in range(1, len(w)):
        a_i = 1.0 - (1.0 - alpha) * gate[i]
        th[i] = a_i * (th[i - 1] + w[i] * dt) + (1.0 - a_i) * tilt[i]
    return th


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


__all__ = ["AnkleAngleConfig", "CALIBRATION_POSE", "sagittal_axis",
           "find_calibration_window", "sagittal_pitch", "ankle_angle",
           "session_calibration", "validate_calibration_pose",
           "verify_foot_side", "stride_profile"]


# ---------------------------------------------------------------------------
# Functional calibration -- measuring the ankle axis instead of assuming it
# ---------------------------------------------------------------------------
#
# Everything above derives each segment's sagittal axis from the segment's own
# motion during walking, then differences two pitches. That is wrong, and it
# fails in a way that looks plausible: the two axes are not the same physical
# axis, so their difference is not a joint angle. Measured on P017 it gave a
# range of motion near 140 degrees against a literature 25-30.
#
# The published fix is Seel, Raisch & Schauer (2014), Sensors 14(4) 6891-6909,
# which solves for one shared hinge axis expressed in both sensor frames by
# minimising  || w1 x j1 || - || w2 x j2 ||  over the recording. It does not
# work here, and not for want of tuning: it assumes a hinge, and the ankle in
# a shoe with strapped sensors is not one. Simulated with a known axis, the
# recovered shank axis degrades to 39 degrees off once 12% of the motion is
# off-axis; this data carries 31-52%.
#
# So do not fit a hinge. Measure each axis from a movement that isolates it,
# which needs no model of the joint at all:
#
#     A   foot rotates, shank still     -> the axis in the FOOT's frame
#     B   shank rotates, foot planted   -> the axis in the SHANK's frame
#     C   both rotate as one rigid body -> the relative sign, and a check
#                                          that A and B found the same axis
#
# Part C matters more than it looks. A and B each recover a direction but not
# a sign, and getting the relative sign wrong adds the two segments' swings
# instead of cancelling them -- which is exactly the 140-degree failure. In a
# rigid swing the two projections must be identical, so their correlation
# settles the sign outright and their amplitude ratio proves both movements
# found the same physical axis.
#
# Against simulation with a known 36.0 degree ankle excursion this recovers
# 36.3 degrees, r = 1.0000, RMSE 0.12 degrees.

FUNCTIONAL_CALIBRATION = (
    "Three movements, about ten seconds each, recorded once per session while "
    "the sensors stay where they are.\n"
    "  A. Sit with the foot off the ground and the lower leg still. Starting "
    "with the foot relaxed, pull the toes up toward the shin and lower them "
    "again, about ten times. Keep the lower leg still -- only the foot moves.\n"
    "  B. Stand with the foot flat and press the heel into the floor. Keeping "
    "the heel down, rock the knee forward over the toes and back, about ten "
    "times. Only the shank moves.\n"
    "  C. Sit with the foot off the ground and hold the ankle stiff, as if it "
    "were in a walking boot. Swing the whole lower leg from the knee about ten "
    "times, so the foot and shank move together as one piece."
)

_PART_NAMES = {"foot": "A (foot moves, shank still)",
               "shank": "B (shank moves, foot planted)",
               "rigid": "C (both move together)"}


def _on_shank_clock(foot: SensorStream, shank: SensorStream):
    """Both segments' gyro on one time base, since the streams are separate."""
    ts = np.asarray(shank.timestamps, dtype=np.float64)
    tf = np.asarray(foot.timestamps, dtype=np.float64)
    if len(ts) < 2 or len(tf) < 2:
        raise ValueError("both streams need at least two samples")
    gs = np.asarray(shank.gyro, dtype=np.float64)
    gf_raw = np.asarray(foot.gyro, dtype=np.float64)
    gf = np.column_stack([np.interp(ts, tf, gf_raw[:, i]) for i in range(3)])
    fs = 1.0 / float(np.median(np.diff(ts)))
    return ts - ts[0], gf, gs, fs


def _longest_run(mask: np.ndarray, gap: int = 3):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return None
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > gap) + 1)
    best = max(runs, key=len)
    return best[0], best[-1]


def find_movement_segments(foot: SensorStream,
                           shank: SensorStream,
                           cfg: AnkleAngleConfig = AnkleAngleConfig(),
                           ) -> Dict[str, Optional[Tuple[float, float]]]:
    """Split a calibration recording into parts A, B and C automatically.

    Classified by which segment is moving, so the parts may be performed in
    any order and the operator does not have to mark them. Returns seconds
    from the start of the shank stream, with None for any part not found.
    """
    t, gf, gs, fs = _on_shank_clock(foot, shank)
    # Smooth over a whole movement cycle, not a fraction of one. Any slow
    # back-and-forth passes through zero velocity at every reversal, so a
    # short window reads those reversals as "still" and shatters one movement
    # into a dozen runs too short to use.
    k = max(int(1.0 * fs), 1)
    box = np.ones(k) / k
    mf = np.convolve(np.linalg.norm(gf, axis=1), box, mode="same")
    ms = np.convolve(np.linalg.norm(gs, axis=1), box, mode="same")

    # Rigid motion is not just "both are moving" -- walking is that too, and
    # walking is usually the longest such stretch in the file. Two sensors on
    # one rigid body measure the same angular velocity vector in different
    # frames, so their magnitudes match; across a real ankle they do not.
    agree = np.abs(mf - ms) < 0.25 * np.maximum(mf, ms)

    want = {
        "foot": (mf > cfg.func_move_dps) & (ms < cfg.func_still_dps),
        "shank": (ms > cfg.func_move_dps) & (mf < cfg.func_still_dps),
        "rigid": (ms > cfg.func_move_dps) & (mf > cfg.func_move_dps) & agree,
    }
    out: Dict[str, Optional[Tuple[float, float]]] = {}
    for name, mask in want.items():
        run = _longest_run(mask, gap=int(0.5 * fs) + 1)
        if run is None or (t[run[1]] - t[run[0]]) < cfg.func_min_s:
            out[name] = None
        else:
            out[name] = (float(t[run[0]]), float(t[run[1]]))
    return out


def _axis_from(g: np.ndarray) -> Tuple[np.ndarray, float]:
    """Dominant rotation axis of a movement, and how strongly it dominates."""
    x = g - g.mean(axis=0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    margin = float(s[0] / max(s[1], 1e-9))
    return vt[0], margin


def functional_calibration(foot: SensorStream,
                           shank: SensorStream,
                           cfg: AnkleAngleConfig = AnkleAngleConfig(),
                           segments: Optional[Dict[str, Tuple[float, float]]] = None,
                           ) -> Dict[str, object]:
    """Measure the ankle axis in both sensor frames from a calibration recording.

    Pass the result to ankle_angle_functional(). Raises if a required part of
    the movement is missing rather than guessing an axis, because a guessed
    axis produces a plausible-looking wrong angle -- see the module note.
    """
    t, gf, gs, fs = _on_shank_clock(foot, shank)
    seg = segments if segments is not None else find_movement_segments(foot, shank, cfg)
    missing = [_PART_NAMES[k] for k in ("foot", "shank", "rigid") if not seg.get(k)]
    if missing:
        raise ValueError(
            "calibration is missing part " + ", ".join(missing) + ". "
            "Record all three movements:\n" + FUNCTIONAL_CALIBRATION)

    def win(name):
        a, b = seg[name]
        return (t >= a) & (t <= b)

    mA, mB, mC = win("foot"), win("shank"), win("rigid")

    # Gyro bias, free of charge: in part A the shank is still, in part B the
    # foot is. A bias of a few deg/s integrates into tens of degrees, so this
    # is not optional.
    bias_s = gs[mA].mean(axis=0)
    bias_f = gf[mB].mean(axis=0)
    gs = gs - bias_s
    gf = gf - bias_f

    j_foot, margin_f = _axis_from(gf[mA])
    j_shank, margin_s = _axis_from(gs[mB])

    # Part C: rigid body, so the two projections must be the same signal.
    # This settles the relative sign, which neither A nor B can do alone.
    pf, ps = gf[mC] @ j_foot, gs[mC] @ j_shank
    corr = float(np.corrcoef(ps, pf)[0, 1]) if len(pf) > 2 else 0.0
    if corr < 0:
        j_foot, pf, corr = -j_foot, -pf, -corr
    gain = float(np.std(pf) / max(np.std(ps), 1e-9))
    # Equal amplitudes are not enough -- two signals can match in size and
    # differ in phase. On a rigid body the two projections are the SAME
    # signal, so what is left after subtracting them is the ankle moving.
    # Walking passes the amplitude test often enough to matter, and fails
    # this one outright.
    residual = float(np.std(ps - pf) / max(np.std(ps), 1e-9))

    # Overall sign, from part B: the heel is down, so the knee can only rock
    # forward from vertical and back. Forward is dorsiflexion, so the shank's
    # excursion away from where it started must come out positive.
    theta_B = np.cumsum(gs[mB] @ j_shank) / fs
    if float(np.mean(theta_B - theta_B[0])) < 0:
        j_shank, j_foot = -j_shank, -j_foot

    # Cross-check on part A. "Relaxed, then toes up" is likewise one-sided, so
    # the foot's excursion should be dorsiflexion too. Advisory only: it rests
    # on the subject having started where they were told to.
    theta_A = np.cumsum(gf[mA] @ j_foot) / fs
    a_ok = bool(float(np.mean(theta_A - theta_A[0])) < 0)

    return dict(
        j_foot=j_foot, j_shank=j_shank,
        bias_foot=bias_f, bias_shank=bias_s,
        segments=seg, fs=float(fs),
        margin_foot=margin_f, margin_shank=margin_s,
        rigid_corr=corr, rigid_gain=gain, rigid_residual=residual,
        part_a_direction_ok=a_ok,
        foot_name=str(foot.name), shank_name=str(shank.name),
    )


def validate_functional_calibration(cal: Dict[str, object],
                                    cfg: AnkleAngleConfig = AnkleAngleConfig(),
                                    ) -> Dict[str, object]:
    """Did the three movements actually identify the axis? Same shape as
    validate_calibration_pose: {'ok', 'warn', 'checks': [...]}."""
    checks = []
    seg = cal["segments"]
    for key in ("foot", "shank", "rigid"):
        w = seg.get(key)
        dur = (w[1] - w[0]) if w else 0.0
        checks.append(dict(
            name=f"part {_PART_NAMES[key][0]} duration",
            level=("ok" if dur >= cfg.func_min_s else "fail"),
            detail=f"{dur:.1f} s of {_PART_NAMES[key]}",
            why=f"needs at least {cfg.func_min_s:.0f} s of clean movement"))

    for key, val in (("foot", cal["margin_foot"]), ("shank", cal["margin_shank"])):
        lvl = "ok" if val >= cfg.func_axis_margin else (
            "warn" if val >= 0.8 * cfg.func_axis_margin else "fail")
        checks.append(dict(
            name=f"{key} axis isolated", level=lvl,
            detail=f"{val:.1f}x dominant",
            why=("the movement has to happen in one plane; a wobbly or "
                 "circling movement identifies no single axis"
                 if lvl != "ok" else "one clear axis of rotation")))

    corr = float(cal["rigid_corr"])
    lvl = "ok" if corr >= cfg.func_corr_min else (
        "warn" if corr >= 0.75 else "fail")
    checks.append(dict(
        name="rigid-swing agreement", level=lvl, detail=f"r = {corr:.3f}",
        why=("in part C the ankle must stay locked so both sensors see the "
             "same rotation; a low value means the ankle moved, and the "
             "relative sign is then not settled"
             if lvl != "ok" else "both sensors saw the same rotation")))

    gain = float(cal["rigid_gain"])
    off = abs(gain - 1.0)
    lvl = "ok" if off <= cfg.func_gain_tol else (
        "warn" if off <= 2 * cfg.func_gain_tol else "fail")
    checks.append(dict(
        name="axes match", level=lvl, detail=f"amplitude ratio {gain:.2f}",
        why=("parts A and B should find the same physical axis, so a rigid "
             "swing projects onto both equally; far from 1 means one of the "
             "two movements was not isolating what it should"
             if lvl != "ok" else "parts A and B found the same axis")))

    resid = float(cal["rigid_residual"])
    lvl = "ok" if resid <= 0.25 else ("warn" if resid <= 0.45 else "fail")
    checks.append(dict(
        name="ankle stayed locked", level=lvl,
        detail=f"{100 * resid:.0f}% of the swing unexplained",
        why=("in part C the two sensors should record the same rotation, so "
             "almost nothing is left when one is subtracted from the other. A "
             "large remainder means the ankle was moving -- most often because "
             "a stretch of walking was picked up as the rigid swing"
             if lvl != "ok" else "the two sensors moved as one body")))

    checks.append(dict(
        name="direction cross-check",
        level=("ok" if cal["part_a_direction_ok"] else "warn"),
        detail=("part A agrees with part B" if cal["part_a_direction_ok"]
                else "part A disagrees with part B"),
        why=("dorsiflexion sign is taken from part B. If part A disagrees, "
             "one of the two movements ran the other way -- check the sign of "
             "the angle before trusting it"
             if not cal["part_a_direction_ok"] else "both parts agree")))

    return dict(ok=not any(c["level"] == "fail" for c in checks),
                warn=any(c["level"] == "warn" for c in checks),
                checks=checks, protocol=FUNCTIONAL_CALIBRATION)


def ankle_angle_functional(foot: SensorStream,
                           shank: SensorStream,
                           calibration: Dict[str, object],
                           heel_strike_times: Optional[np.ndarray] = None,
                           cfg: AnkleAngleConfig = AnkleAngleConfig(),
                           ) -> Dict[str, object]:
    """Sagittal ankle angle using a measured axis. Dorsiflexion positive.

    Returns the joint angular velocity always, and a stride-normalised angle
    profile when heel strikes are supplied. There is no continuous angle
    trace: integrating one across a whole trial accumulates bias drift with
    nothing to pin it down, so the angle is only ever integrated within a
    stride and referenced to that stride's own mean.
    """
    t, gf, gs, fs = _on_shank_clock(foot, shank)
    gf = gf - np.asarray(calibration["bias_foot"], dtype=np.float64)
    gs = gs - np.asarray(calibration["bias_shank"], dtype=np.float64)
    j_f = np.asarray(calibration["j_foot"], dtype=np.float64)
    j_s = np.asarray(calibration["j_shank"], dtype=np.float64)

    # Dorsiflexion closes the ankle, which the shank does by rocking forward
    # and the foot by lifting its toes -- opposite senses about the same axis,
    # so the joint rate is the difference of the two projections.
    rate = (gs @ j_s) - (gf @ j_f)
    out: Dict[str, object] = dict(t=t, rate=rate, fs=float(fs))
    if heel_strike_times is None:
        return out

    hs = np.sort(np.asarray(heel_strike_times, dtype=np.float64))
    grid = np.linspace(0.0, 1.0, 101)
    strides = []
    for a, b in zip(hs[:-1], hs[1:]):
        dur = b - a
        if not (0.7 < dur < 2.0):
            continue
        m = (t >= a) & (t < b)
        n = int(m.sum())
        if n < 40:
            continue
        ang = np.cumsum(rate[m]) / fs
        # The ankle returns to where it started each stride, so any residual
        # ramp is drift, not motion. Removing it costs nothing real.
        ang = ang - np.linspace(ang[0], ang[-1], n)
        ang = ang - ang.mean()
        strides.append(np.interp(grid, np.linspace(0.0, 1.0, n), ang))

    if not strides:
        out.update(n_strides=0, profile=None, profile_sd=None, rom=float("nan"))
        return out
    P = np.asarray(strides)
    mu = P.mean(axis=0)
    out.update(n_strides=len(P), profile=mu, profile_sd=P.std(axis=0),
               rom=float(np.ptp(mu)),
               dorsiflexion_peak=float(mu.max()),
               plantarflexion_peak=float(mu.min()),
               plantarflexion_at=int(np.argmin(mu)))
    return out
