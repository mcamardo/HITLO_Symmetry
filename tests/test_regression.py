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
    cfg = yaml.safe_load((REPO / 'config' / 'exo_symmetry_config.yml').read_text())
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
    cfg = yaml.safe_load((REPO / 'config' / 'exo_symmetry_config.yml').read_text())
    table = IndexTable(str(REPO / cfg['Optimization']['index_csv']))
    doses = [table.row(float(v))['dose_Nm'] for v in cfg['Optimization']['ramp_sequence']]
    full = table.df['dose_signed_Nm']
    span = float(full.max() - full.min())
    covered = (max(doses) - min(doses)) / span
    assert covered > 0.90, f"ramp covers only {covered*100:.0f}% of the dose range"
    assert min(doses) < 0 and max(doses) > 0, "ramp must touch both arms"
    return f"ramp covers {covered*100:.0f}% of dose range, {min(doses):+.1f} to {max(doses):+.1f} Nm"


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
