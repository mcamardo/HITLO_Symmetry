#!/usr/bin/env python3.12
"""
tests/test_regression.py — guards against silent-failure bugs.

Run with plain python (no pytest needed):

    python3.12 tests/test_regression.py

Every test here corresponds to a bug that actually shipped and cost session
time. The common thread is that none of them raised an error — they returned
plausible-looking wrong answers, or nothing at all.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from hitlo.detection import (DetectionConfig, compute_magnitude,
                             detect_heelstrikes_full)
from hitlo.symmetry import compute_symmetry_index
from hitlo.index_unified import IndexTable

# A HISTORICAL recording, deliberately not one from the current session.
# This pointed at sub-P998 Pre_run-002 and broke the moment that filename was
# re-recorded with a bad trial -- a regression fixture must not live in a
# directory the experiment writes to.
REAL_XDF = Path("/Users/maccamardo/HITLO_Data/sub-P062/ses-S001/eeg/"
                "sub-P062_ses-S001_task-Pre_run-002_eeg.xdf")
EXPECTED_SI = -15.46   # established value for this recording


def _synthetic_gait(n=8000, fs=200.0, step_s=0.6, amp=3000.0):
    """Impulse train on a 1 g baseline — a crude but detectable gait signal."""
    t = np.arange(n) / fs
    mag = np.full(n, 1000.0)
    for k in range(int(n / fs / step_s)):
        i = int(k * step_s * fs)
        if i + 6 < n:
            mag[i:i + 6] += amp * np.hanning(6)
    a = np.zeros((n, 3))
    a[:, 2] = mag
    return a, t + 1000.0


# ---------------------------------------------------------------------------

def _load_config():
    """The live config if present, else the tracked example.

    exo_symmetry_config.yml is gitignored personal state, so a fresh clone --
    or CI -- will not have it. Falling back to the example keeps these tests
    runnable by anyone, and stops the suite from silently depending on one
    machine's settings.
    """
    import yaml
    for name in ('exo_symmetry_config.yml', 'exo_symmetry_config.example.yml'):
        p = REPO / 'config' / name
        if p.exists():
            return yaml.safe_load(p.read_text())
    raise FileNotFoundError(
        "no config found: expected config/exo_symmetry_config.yml or "
        "config/exo_symmetry_config.example.yml")


def test_dtype_invariance():
    """int16 accel must give byte-identical results to float.

    THE BUG: the LSL outlet declares channel_format='int16', so pyxdf returns
    int16. compute_magnitude squared it, int16 overflowed, sqrt(negative) = NaN,
    and every downstream threshold silently compared False. Result: 0 heel
    strikes on a perfect 119s recording, reported only as "Cost extraction
    failed". Older float32 files masked it for months.
    """
    a_f, ts = _synthetic_gait()
    ref = detect_heelstrikes_full(a_f, ts, cfg=DetectionConfig())
    n_ref = len(ref.heel_strike_indices)
    assert n_ref > 10, f"synthetic signal should yield strikes, got {n_ref}"

    for dt in (np.int16, np.int32, np.float32, np.float64):
        got = detect_heelstrikes_full(a_f.astype(dt), ts, cfg=DetectionConfig())
        n = len(got.heel_strike_indices)
        assert n == n_ref, f"dtype {np.dtype(dt).name}: {n} strikes vs {n_ref} for float64"
    return f"{n_ref} strikes, identical across int16/int32/float32/float64"


def test_magnitude_rejects_garbage():
    """Non-finite or wrong-shaped input must raise, never return silently."""
    for label, arr in (("NaN", np.array([[np.nan, 1, 2]] * 5)),
                       ("inf", np.array([[np.inf, 1, 2]] * 5)),
                       ("(N,2)", np.zeros((5, 2)))):
        try:
            compute_magnitude(arr)
        except ValueError:
            continue
        raise AssertionError(f"{label} input did not raise")
    return "NaN, inf, and wrong-shape all raise ValueError"


def test_symmetry_sign_convention():
    """SI > 0 must mean right step longer than left (hitlo/symmetry.py)."""
    right = np.full(20, 0.70)
    left = np.full(20, 0.60)
    si, _ = compute_symmetry_index(right, left, signed=True)
    assert si > 0, f"right>left should give SI>0, got {si:+.2f}"
    si2, _ = compute_symmetry_index(left, right, signed=True)
    assert si2 < 0, f"left>right should give SI<0, got {si2:+.2f}"
    assert abs(si + si2) < 1e-9, "sign convention is not antisymmetric"
    return f"right-longer=+{si:.2f}%, left-longer={si2:.2f}%"


def test_index_table_ramp_snaps_exactly():
    """Every configured ramp x must be a real table row, not an interpolation.

    Interpolating between rows yields a configuration that never passed the
    builder's safety filters.
    """
    import yaml
    cfg = _load_config()
    table = IndexTable(str(REPO / cfg['Optimization']['index_csv']))
    for x in cfg['Optimization']['ramp_sequence']:
        snapped = table.snap(float(x))
        assert abs(snapped - float(x)) < 1e-9, \
            f"ramp x={x} snaps to {snapped} — not a real row"
    assert abs(table.x_values[0] + 1.0) < 1e-9
    assert abs(table.x_values[-1] - 1.0) < 1e-9
    return f"{len(cfg['Optimization']['ramp_sequence'])} ramp values all exact rows"


def test_acc_request_has_no_channel_field():
    """The H10 rejects an ACC config carrying a channel-count setting.

    THE BUG: ACC_WRITE ended with 0x04,0x01,0x03. The sensor answered
    f0 02 02 05 (ERROR INVALID PARAMETER), the script never read that response,
    and it printed success while opening an LSL outlet that carried no data.
    """
    import re
    src = (REPO / 'apps' / 'collect_sensors.py').read_text()
    m = re.search(r'ACC_WRITE = bytearray\(\[(.*?)\]\)', src, re.S)
    assert m, "ACC_WRITE not found"
    body = m.group(1)
    vals = [int(v, 16) for v in re.findall(r'0x([0-9A-Fa-f]{2})', body)]
    assert 0x04 not in vals[1:], \
        "ACC_WRITE contains a channel-count setting (0x04) — H10 rejects it"
    assert 'PMD_ERRORS' in src, "control-response check was removed"
    return f"payload is {len(vals)} bytes, no channel field, response checked"


def test_bo_axis_is_stiffness_not_rank():
    """BO must search normalized stiffness, not index rank.

    THE PROBLEM: x is a rank. The DF arm is 15 of 46 rows (33% of the x axis)
    but spans only 16.5 of 260 Nm/rad (5.6% of the achievable torque). A GP
    searching x therefore spent a third of its trials on an arm that had almost
    nothing left to give — observed in sub-P997, where 6 of 15 trials went to
    dorsiflexion and BO kept returning there.
    """
    table = IndexTable(str(REPO / 'config' / 'index_unified.csv'))
    u = table.u_values
    assert np.all(np.diff(u) > 0), "u must be strictly increasing to be invertible"
    assert abs(u[0]) < 1e-12 and abs(u[-1] - 1.0) < 1e-12, "u must span [0, 1]"

    x = table.x_values
    assert np.allclose(table.x_of_u(table.u_of(x)), x), "u<->x round trip must be exact"

    df_frac = float((table.df['direction'] < 0).mean())
    df_span = float(u[table.df['direction'].to_numpy() < 0].max())
    assert df_frac > 0.25, "sanity: DF really is a large share of the rank axis"
    # The window is deliberate and bounded on BOTH sides.
    #  - Above ~0.31 (the rank axis) BO over-explores an arm that has little
    #    torque left to give: 6 of 15 trials went to DF in sub-P997.
    #  - Below ~0.10 the arm is narrower than the GP's 0.05 lengthscale floor
    #    and BO cannot resolve WITHIN it. Verified with linear stiffness
    #    (DF = 5.6%): a synthetic optimum at DF max was never found, BO
    #    stalling at x=-0.53. Signed sqrt puts DF at ~0.14, where a planted
    #    DF optimum IS found (x=-0.93) and a PF optimum still draws 0 DF trials.
    assert 0.10 < df_span < 0.20, (
        f"DF spans {df_span:.3f} of the u axis; outside [0.10, 0.20] it is "
        f"either unresolvable or over-weighted")
    return (f"DF is {df_frac*100:.0f}% of rank axis, {df_span*100:.1f}% of "
            f"search axis (resolvable, not over-weighted)")


def test_ramp_spans_the_torque_range():
    """The manual ramp must exercise the device's real range.

    The original ramp (0, ±0.2, ±0.4) never applied more than 17.7 Nm of the
    47.1 Nm available and covered 38% of the achievable dose range, so the GP
    entered BO having never seen most of what the device can do.
    """
    import yaml
    cfg = _load_config()
    table = IndexTable(str(REPO / cfg['Optimization']['index_csv']))
    doses = [table.row(float(v))['dose_Nm'] for v in cfg['Optimization']['ramp_sequence']]
    full = table.df['dose_signed_Nm']
    span = float(full.max() - full.min())
    covered = (max(doses) - min(doses)) / span
    assert covered > 0.90, f"ramp covers only {covered*100:.0f}% of the dose range"
    assert min(doses) < 0 and max(doses) > 0, "ramp must touch both arms"
    return f"ramp covers {covered*100:.0f}% of dose range, {min(doses):+.1f} to {max(doses):+.1f} Nm"


def test_detection_config_tracks_the_real_sample_rate():
    """cfg.fs must follow the hardware, not a hardcoded default.

    Every window in the detector is written in seconds and converted with
    fs, so a wrong fs rescales the lowpass cutoff, the minimum peak
    separation, the cluster gap and the stance window simultaneously. That
    is invisible for Polar (nominal 200, measured 199.6-201.3) and a 26%
    error for a Trigno Avanti at 148 Hz — detection would still return
    plausible heel strikes, at the wrong times.
    """
    import warnings as _w

    class _S:
        actual_fs = 148.1

    base = DetectionConfig()
    assert base.fs == 200, "default should still describe the Polar hardware"

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        cfg = base.for_stream(_S())
        assert cfg.fs == 148, f"expected 148, got {cfg.fs}"
        assert caught, "a >5% sample-rate mismatch must warn, not pass silently"

    # A rate close to the default must NOT warn, or the warning becomes noise
    # that gets ignored on the day it matters.
    class _P:
        actual_fs = 200.6

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        cfg_p = base.for_stream(_P())
        assert cfg_p.fs == 201
        assert not caught, "a sub-1% difference should be silent"

    # The windows must actually move with fs.
    assert int(base.cluster_gap_s * cfg.fs) < int(base.cluster_gap_s * base.fs)
    return f"148.1 Hz -> fs=148 with warning; 200.6 Hz -> fs=201 silently"


def test_polar_backend_unchanged_by_io_generalization():
    """Generalizing io for Trigno must not perturb the Polar path.

    load_streams with no config, and with backend=polar, must both return
    exactly what load_both_polar_streams returned before the refactor.
    """
    from hitlo.io import (load_both_polar_streams, load_streams,
                          live_stream_names, PolarStream, SensorStream)
    if not REAL_XDF.is_file():
        return "SKIPPED — reference recording not on this machine"
    a, _ = load_both_polar_streams(str(REAL_XDF))
    b, _ = load_streams(str(REAL_XDF), None)
    c, _ = load_streams(str(REAL_XDF), {'Sensing': {'backend': 'polar'}})
    assert a is not None and b is not None and c is not None
    assert np.array_equal(a.accel, b.accel) and np.array_equal(a.accel, c.accel)
    assert np.array_equal(a.timestamps, b.timestamps)
    assert PolarStream is SensorStream, "old name must still resolve"
    assert a.gyro is None and not a.has_gyro, "Polar has no gyro"
    assert live_stream_names(None) == ['polar accel left', 'polar accel right']
    assert live_stream_names({'Sensing': {'backend': 'trigno'}}) == ['TrignoIMU']
    return "polar load identical through the dispatch layer"


def _synth_shank_gyro(fs=148.0, stride=1.20, n_strides=30, contact_frac=0.72,
                      swing_amp=300.0, stance_frac=0.35, noise=4.0,
                      sign=+1, width=0.12, seed=0):
    """Shank sagittal velocity with KNOWN contact times.

    Big positive swing peak, zero crossing exactly at contact, smaller
    negative stance lobe after — the shape the zero-crossing rule keys on.
    """
    rng = np.random.default_rng(seed)
    n = int(n_strides * stride * fs)
    t = np.arange(n) / fs
    w = np.zeros(n)
    contacts = []
    for k in range(n_strides):
        tc = k * stride + contact_frac * stride
        if tc > t[-1] - 0.6 or tc < 0.6:
            continue
        contacts.append(tc)
        m = np.abs(t - tc) < 0.45
        tau = t[m] - tc
        lobe = (-tau / width) * np.exp(-0.5 * (tau / width) ** 2) * np.e ** 0.5
        lobe = np.where(tau > 0, lobe * stance_frac, lobe)
        w[m] += swing_amp * lobe
    w += rng.normal(0, noise, n)
    g = np.zeros((n, 3))
    g[:, 2] = w * sign
    return g, t + 1000.0, np.array(contacts) + 1000.0


def test_gyro_detector_finds_known_contacts():
    """Zero-crossing detection must recover contacts it was given."""
    from hitlo.detection_gyro import GyroDetectionConfig, detect_heelstrikes_gyro
    worst = 0.0
    for label, kw in (("nominal", {}),
                      ("noisy", {"noise": 16.0}),
                      ("slow", {"stride": 1.6}),
                      ("fast", {"stride": 0.95}),
                      ("damped swing", {"swing_amp": 100.0}),
                      ("200 Hz", {"fs": 200.0})):
        fs = kw.get("fs", 148.0)
        g, t, truth = _synth_shank_gyro(**kw)
        res = detect_heelstrikes_gyro(
            g, t, cfg=GyroDetectionConfig(fs=int(fs)))
        det = np.asarray(res.heel_strike_times)
        assert len(det) == len(truth), (
            f"{label}: found {len(det)} of {len(truth)} contacts")
        err = np.array([abs(det[np.argmin(np.abs(det - c))] - c) * 1000
                        for c in truth])
        assert np.median(err) < 10.0, (
            f"{label}: median timing error {np.median(err):.1f} ms")
        worst = max(worst, float(np.median(err)))
    return f"all contacts recovered in 6 regimes, worst median error {worst:.1f} ms"


def test_session_calibration_reuse_is_exact_and_guarded():
    """Capturing a calibration once and reusing it must reproduce the direct
    path exactly, and must refuse a sensor it did not come from.

    Standing still before every trial is not practical, so a session-level
    calibration is the realistic workflow. Measured on a recording with two
    quiet stands 105 s apart, reuse costs a 0.9 degree offset with the
    stride-averaged shape correlating at r = 0.9999 -- but only while the
    sensors stay put. A zero encodes where a sensor sits on the limb, so
    applying one sensor's calibration to another gives a smooth, plausible,
    wrong angle. That has to raise, not warn.
    """
    from hitlo.ankle_angle import ankle_angle, session_calibration
    from hitlo.io import SensorStream
    fs, n = 148.0, int(60 * 148)
    t = np.arange(n) / fs
    rng = np.random.default_rng(5)

    def limb(name, bias, tilt_deg):
        w = 200.0 * np.sin(2 * np.pi * t / 1.2) + bias
        w[t < 12] = bias + rng.normal(0, 0.5, int((t < 12).sum()))   # quiet stand
        g = np.zeros((n, 3)); g[:, 2] = w
        a = np.zeros((n, 3))
        a[:, 0] = np.sin(np.radians(tilt_deg)); a[:, 1] = np.cos(np.radians(tilt_deg))
        a += rng.normal(0, .01, a.shape)
        return SensorStream(name=name, timestamps=t + 1000.0, accel=a, gyro=g,
                            actual_fs=fs)

    foot = limb("foot", -8.9, 20.0)
    shank = limb("shank", 2.1, 3.0)

    cal = session_calibration(foot, shank, calib=(1.0, 11.0))
    direct = ankle_angle(foot, shank, calib=(1.0, 11.0))["angle"]
    reused = ankle_angle(foot, shank, calibration=cal)["angle"]
    worst = float(np.max(np.abs(direct - reused)))
    assert worst < 1e-9, f"reuse differs from the direct path by {worst:.3e} deg"

    stranger = limb("someone_else", 2.1, 3.0)
    try:
        ankle_angle(foot, stranger, calibration=cal)
    except ValueError:
        pass
    else:
        raise AssertionError("a calibration from another sensor was accepted")
    return f"reuse exact to {worst:.1e} deg; mismatched sensor refused"


def test_gyro_axis_is_measured_not_assumed():
    """The sagittal axis identifies itself; a pinned wrong axis warns.

    This defaulted to z, which is correct for a shank sensor mounted the way
    ours happen to be and wrong for anything else -- a foot sensor in the same
    recording resolves to x. The failure is silent: the detector reads a
    mostly-flat channel, finds peaks in noise, and returns events that look
    like clean periodic detection.

    Builds the same walking signal on each of the three axes in turn and
    requires identical contact times from all three.
    """
    import warnings as _w
    from hitlo.detection_gyro import GyroDetectionConfig, detect_heelstrikes_gyro
    cfg = GyroDetectionConfig(fs=148)
    base, t, truth = _synth_shank_gyro(stance_frac=0.35)
    w = base[:, 2].copy()
    rng = np.random.default_rng(11)
    found = {}
    for ax in (0, 1, 2):
        g = rng.normal(0, 3.0, base.shape)      # quiet noise on the other axes
        g[:, ax] = w
        det = np.asarray(detect_heelstrikes_gyro(g, t, cfg=cfg).heel_strike_times)
        assert len(det) == len(truth), (
            f"signal on axis {'xyz'[ax]}: found {len(det)} of {len(truth)}")
        err = np.median([abs(det[np.argmin(np.abs(det - c))] - c) * 1000
                         for c in truth])
        assert err < 10.0, f"axis {'xyz'[ax]}: {err:.1f} ms from truth"
        found['xyz'[ax]] = err
    # pinning the wrong axis must complain rather than fail quietly
    g = rng.normal(0, 3.0, base.shape); g[:, 0] = w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        detect_heelstrikes_gyro(g, t, cfg=GyroDetectionConfig(fs=148, sagittal_axis=1))
    assert any("sagittal_axis" in str(c.message) for c in caught), (
        "pinning the wrong sagittal axis produced no warning")
    return (f"found on all three axes (worst {max(found.values()):.1f} ms); "
            f"wrong pin warns")


def test_foot_sensor_side_check_catches_a_swap():
    """A foot swings with the shank above it, not the one across from it.

    Guards a real labelling failure: a sensor recorded as left_foot was
    physically on the right foot for a whole session. Pairing on the label
    computes the angle between two segments that never move together, which
    comes out large, smooth, and completely wrong -- exactly the kind of
    result that looks like a finding.

    Synthesises two shanks half a stride apart and a foot locked to one of
    them, then checks the phase test picks the right leg either way.
    """
    from hitlo.ankle_angle import verify_foot_side
    from hitlo.io import SensorStream
    fs, stride, n = 148.0, 1.2, int(40 * 1.2 * 148)
    t = np.arange(n) / fs
    rng = np.random.default_rng(3)

    def limb(phase, amp=250.0):
        w = amp * np.sin(2 * np.pi * (t / stride - phase))
        w += 0.35 * amp * np.sin(4 * np.pi * (t / stride - phase))
        g = np.zeros((n, 3)); g[:, 2] = w + rng.normal(0, 4, n)
        a = np.zeros((n, 3)); a[:, 1] = 1.0 + rng.normal(0, .02, n)
        return SensorStream(name="s", timestamps=t + 1000.0, accel=a, gyro=g,
                            actual_fs=fs)

    left, right = limb(0.0), limb(0.5)
    for truth, same, other in (("left", left, right), ("right", right, left)):
        foot = limb(0.0 if truth == "left" else 0.5, amp=320.0)
        ok = verify_foot_side(foot, same, other)
        assert ok["agrees"], (
            f"foot on {truth} not recognised: lag to own shank "
            f"{ok['lag_same_s']*1000:.0f} ms vs other {ok['lag_other_s']*1000:.0f} ms")
        swapped = verify_foot_side(foot, other, same)
        assert not swapped["agrees"], (
            f"a swapped pairing was accepted for the {truth} foot")
    return "correct pairing accepted, swapped pairing rejected, both legs"


def test_gyro_polarity_prefers_the_larger_lobe():
    """Mid-swing is the fastest rotation in the cycle, so it is the larger lobe.

    This is the regression guard for a real bug. The detector used to choose
    polarity by which orientation produced MORE events, tying on regularity.
    Both orientations of a periodic signal give similarly many, similarly
    regular events, so that criterion cannot discriminate -- and the count
    tiebreak is backwards, because the WRONG lobe yields more events (the
    stance reversal contributes extra zero crossings). Measured against
    accelerometer heel strikes on sub-P012/ses-S001 it picked the wrong lobe
    on 16 of 16 legs, putting every event ~480 ms early: near toe-off rather
    than contact. Step-time symmetry survived that (a shared offset cancels)
    but every gait-phase result built on it was a third of a cycle wrong.
    """
    from hitlo.detection_gyro import GyroDetectionConfig, detect_heelstrikes_gyro
    cfg = GyroDetectionConfig(fs=148)
    # stance_frac 0.35 -> swing lobe is ~3x the stance lobe, as in real shank
    # data (measured ratios 1.14-1.82). Feed both mountings; contact times
    # must come back the same either way.
    got = {}
    for sign in (+1, -1):
        g, t, truth = _synth_shank_gyro(sign=sign, stance_frac=0.35)
        det = np.asarray(detect_heelstrikes_gyro(g, t, cfg=cfg).heel_strike_times)
        assert len(det) == len(truth), (
            f"sign={sign:+d}: found {len(det)} of {len(truth)}")
        err = np.median([abs(det[np.argmin(np.abs(det - c))] - c) * 1000
                         for c in truth])
        # A wrong-lobe lock lands a third of a stride away (~400 ms at
        # stride 1.2 s), so this bound separates the two outcomes by 40x.
        assert err < 10.0, f"sign={sign:+d}: {err:.1f} ms from truth"
        got[sign] = err
    return (f"larger lobe wins in both mountings, worst {max(got.values()):.1f} ms")


def test_gyro_detector_survives_inverted_mounting():
    """Gyro polarity depends on how the sensor was clipped on.

    Getting it wrong locks onto the stance reversal instead of swing: every
    event lands at the wrong point in the cycle while still looking like a
    clean periodic detection. Inferring polarity from which excursion is
    larger fails when the two lobes are comparable — it decides on noise —
    so the detector runs BOTH polarities and keeps the better result.
    """
    from hitlo.detection_gyro import GyroDetectionConfig, detect_heelstrikes_gyro
    cfg = GyroDetectionConfig(fs=148)
    out = {}
    for sign in (+1, -1):
        for stance_frac, tag in ((0.35, "asymmetric"), (1.0, "symmetric")):
            g, t, truth = _synth_shank_gyro(sign=sign, stance_frac=stance_frac)
            det = np.asarray(
                detect_heelstrikes_gyro(g, t, cfg=cfg).heel_strike_times)
            assert len(det) == len(truth), (
                f"sign={sign:+d} {tag}: found {len(det)} of {len(truth)}")
            err = np.median([abs(det[np.argmin(np.abs(det - c))] - c) * 1000
                             for c in truth])
            assert err < 10.0, f"sign={sign:+d} {tag}: {err:.1f} ms error"
            out[(sign, tag)] = err
    return ("polarity resolved in all 4 combinations, worst "
            f"{max(out.values()):.1f} ms")


def test_gyro_peak_spacing_adapts_to_cadence():
    """Swing-peak spacing must follow the subject's stride, not a fixed floor.

    THE BUG: min_peak_dist_s defaulted to 0.40 s, chosen against synthetic
    data. On a real 1.42 s stride that admits three "swing peaks" per cycle,
    so post-contact ringing on one limb became a second detected contact.
    Observed on sub-P012: one leg returned 62 events to the other's 56 over
    the same window, alternation fell to 87%, and SEM was 4x worse.

    The detector now runs a second pass with spacing derived from the stride
    the first pass measured.
    """
    from hitlo.detection_gyro import GyroDetectionConfig, detect_heelstrikes_gyro
    from dataclasses import replace

    # A long stride with a ringing artifact after each contact — the shape
    # that defeats a fixed floor.
    fs, stride = 148.0, 1.45
    g, t, truth = _synth_shank_gyro(fs=fs, stride=stride, n_strides=30)
    w = g[:, 2].copy()
    rng = np.random.default_rng(3)
    # A spurious swing-peak-then-crossing partway through stance. It has to
    # sit FURTHER than the 0.40 s floor from the real swing peak (or the floor
    # suppresses it and the fixture proves nothing) but closer than
    # 0.6 x stride, which is what the adaptive pass rejects it by.
    for c in truth:
        i = int((c - t[0] + 0.55) * fs)      # 0.55 s into stance
        k = np.arange(-int(0.16 * fs), int(0.16 * fs))
        if i + k[0] < 0 or i + k[-1] >= len(w):
            continue
        tau = k / fs
        w[i + k] += 190.0 * (-tau / 0.07) * np.exp(-0.5 * (tau / 0.07) ** 2) * np.e ** 0.5
    g[:, 2] = w + rng.normal(0, 3, len(w))

    base = GyroDetectionConfig(fs=int(fs))
    fixed = detect_heelstrikes_gyro(
        g, t, cfg=replace(base, adaptive_dist_frac=None))
    adaptive = detect_heelstrikes_gyro(g, t, cfg=base)

    n_fix = len(fixed.heel_strike_times)
    n_ada = len(adaptive.heel_strike_times)
    assert n_fix > len(truth), (
        "the fixture should over-detect with a fixed floor, or it is not "
        f"exercising the bug (got {n_fix} for {len(truth)} contacts)")
    assert abs(n_ada - len(truth)) <= 2, (
        f"adaptive pass should recover ~{len(truth)} contacts, got {n_ada}")

    def cv(x):
        iv = np.diff(np.sort(x))
        return float(np.std(iv) / np.mean(iv))

    assert cv(adaptive.heel_strike_times) < cv(fixed.heel_strike_times), (
        "adaptive spacing should produce a more regular event series")
    return (f"fixed floor {n_fix} events (CV {cv(fixed.heel_strike_times):.3f}), "
            f"adaptive {n_ada} (CV {cv(adaptive.heel_strike_times):.3f}), "
            f"truth {len(truth)}")


def test_gyro_timing_bias_is_common_mode():
    """A shared timing offset must cancel out of the symmetry index.

    Filter group delay shifts BOTH legs' events by the same amount. Step
    time is the gap between a left and a right event, so a common offset
    cancels exactly and only a DIFFERENTIAL bias can reach SI. This pins
    that, because if it ever stopped being true the detector would inject
    asymmetry that looks like gait.
    """
    from hitlo.detection_gyro import GyroDetectionConfig, detect_heelstrikes_gyro
    from hitlo.symmetry import compute_step_times, compute_symmetry_index
    cfg = GyroDetectionConfig(fs=148)
    gl, tl, _ = _synth_shank_gyro(contact_frac=0.72, seed=1)
    gr, tr, _ = _synth_shank_gyro(contact_frac=0.22, seed=2)
    lt = np.asarray(detect_heelstrikes_gyro(gl, tl, cfg=cfg).heel_strike_times)
    rt = np.asarray(detect_heelstrikes_gyro(gr, tr, cfg=cfg).heel_strike_times)
    rs, ls = compute_step_times(lt, rt)
    n = min(len(rs), len(ls))
    assert n >= 10, f"only {n} stride pairs from the synthetic pair"
    si, _ = compute_symmetry_index(rs[:n], ls[:n], signed=True)
    # Both legs are the same synthetic waveform offset in phase, so any
    # detector bias is identical on both and must not create asymmetry.
    assert abs(si) < 6.0, (
        f"identical waveforms on both legs produced SI={si:+.2f}%, so the "
        f"detector is injecting asymmetry rather than measuring it")
    return f"identical L/R waveforms give SI={si:+.2f}% (bias cancels)"


TRIGNO_DIR = Path("/Users/maccamardo/HITLO_Data/sub-P012/ses-S001/motion")


def test_both_backends_run_end_to_end():
    """Polar and Trigno must both work through the same cost function.

    The point of the migration was that the optimizer should not care which
    sensor produced the events. This pins that: same SymmetryCost, two
    configs, two file layouts (eeg/ vs motion/), two detectors.
    """
    from hitlo.cost import SymmetryCost
    from hitlo.io import trial_dir, trial_filename, backend_modality

    results = {}

    # Polar — must still return exactly the historical value.
    pc = {'Subject': {'id': 'P062', 'session': 'S001',
                      'base_dir': '/Users/maccamardo/HITLO_Data'},
          'Sensing': {'backend': 'polar'}, 'Cost': {}}
    if REAL_XDF.is_file():
        c = SymmetryCost(trial_data_dir=str(trial_dir(pc)), subject_id='P062',
                         session='S001', signed=True, si_target=-3.0,
                         trim_seconds=3.0, config=pc)
        a = c.analyze_trial(
            trial_num=2,
            filename=trial_filename('P062', 'S001', 2, task='Pre',
                                    modality=backend_modality(pc)),
            verbose=False)
        assert a is not None, f"polar path failed: {c.last_failure}"
        assert abs(a.symmetry_index - EXPECTED_SI) < 0.5, (
            f"polar SI drifted to {a.symmetry_index:+.2f}%")
        results['polar'] = a.symmetry_index

    # Trigno — the validation set, where the manipulation is known.
    tc = {'Subject': {'id': 'P012', 'session': 'S001',
                      'base_dir': '/Users/maccamardo/HITLO_Data'},
          'Sensing': {'backend': 'trigno', 'detector': 'gyro',
                      'stream': 'TrignoIMU'}, 'Cost': {}}
    if not TRIGNO_DIR.is_dir():
        if not results:
            return "SKIPPED — no reference recordings on this machine"
        return f"polar {results['polar']:+.2f}%; trigno recordings absent"

    c = SymmetryCost(trial_data_dir=str(trial_dir(tc)), subject_id='P012',
                     session='S001', signed=True, si_target=-3.0,
                     trim_seconds=3.0, config=tc)
    got = {}
    for run in (2, 7, 8):
        fn = trial_filename('P012', 'S001', run,
                            modality=backend_modality(tc))
        if not (TRIGNO_DIR / fn).is_file():
            continue
        a = c.analyze_trial(trial_num=run, filename=fn, verbose=False)
        assert a is not None, f"trigno run {run} failed: {c.last_failure}"
        got[run] = a.symmetry_index

    # The known manipulation: run 7 limped on the right, run 8 on the left.
    # Limping right shortens right stance, so the left foot lands sooner
    # after the right -> left step SHORT, right step LONG -> SI positive.
    if {2, 7, 8} <= set(got):
        assert got[7] > got[2], (
            f"right-limp trial should exceed baseline: "
            f"{got[7]:+.2f}% vs {got[2]:+.2f}%")
        assert got[8] < got[2], (
            f"left-limp trial should fall below baseline: "
            f"{got[8]:+.2f}% vs {got[2]:+.2f}%")
        return (f"polar {results.get('polar', float('nan')):+.2f}%; trigno "
                f"normal {got[2]:+.2f}%, limp-R {got[7]:+.2f}%, "
                f"limp-L {got[8]:+.2f}% (signs correct)")
    return f"trigno ran on {len(got)} recording(s)"


def test_real_recording_end_to_end():
    """The actual file that failed today must now analyze cleanly."""
    if not REAL_XDF.is_file():
        return "SKIPPED — reference recording not on this machine"
    from hitlo.cost import SymmetryCost
    c = SymmetryCost(trial_data_dir=str(REAL_XDF.parent), subject_id="P998",
                     session="S001", signed=True, si_target=-3.0, trim_seconds=3.0)
    a = c.analyze_trial(trial_num=2, filename=REAL_XDF.name, verbose=False)
    assert a is not None, f"analysis returned None: {c.last_failure}"
    assert len(a.left_heel_strikes) > 30, "implausibly few left heel strikes"
    assert len(a.right_heel_strikes) > 30, "implausibly few right heel strikes"
    assert abs(a.symmetry_index - EXPECTED_SI) < 0.5, (
        f"SI drifted: {a.symmetry_index:+.2f}% vs expected {EXPECTED_SI:+.2f}% "
        f"— a pipeline change altered the result on a fixed recording")
    return (f"L={len(a.left_heel_strikes)} R={len(a.right_heel_strikes)} "
            f"SI={a.symmetry_index:+.2f}%")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    print(f"Running {len(tests)} regression tests\n")
    for t in tests:
        try:
            detail = t()
            print(f"  PASS  {t.__name__}\n        {detail}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}\n        {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {t.__name__}\n        {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
