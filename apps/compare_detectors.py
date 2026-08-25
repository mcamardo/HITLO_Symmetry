#!/usr/bin/env python3.12
"""
apps/compare_detectors.py — run both detectors over one recording.

    ./apps/compare_detectors.py <file.xdf>
    ./apps/compare_detectors.py <file.xdf> --plot out.png

Requires a recording that carries BOTH accelerometer and gyroscope, i.e. a
Trigno file. Polar recordings have no gyro, so there is nothing to compare.

WHY THIS EXISTS
---------------
Switching detection method changes which instant is called "heel strike":
the gyro zero crossing is initial contact, the accelerometer jerk peak is
the impact shock that follows it. Two things follow, and this tool measures
both on the same walk so neither has to be argued about:

1. How far apart the two methods place contact. A constant offset is
   harmless for symmetry index -- step time is a difference between legs, so
   a shared offset cancels. A DIFFERENT offset per leg does not cancel: it
   moves SI by 0.4/stride points per millisecond, which the tool computes
   from the recording's own stride time rather than assuming a value.

2. Whether the accelerometer path was mismeasuring one leg. That is the open
   question from the Polar sessions: a persistent ~74 ms left-late offset
   that survived swapping the sensors between legs and was not explained by
   missed strikes, candidate density, timestamps, clock drift, or the SI
   arithmetic. If it was an artifact of choosing among ambiguous jerk peaks
   on the damped limb, the gyro path is immune to it and the two SI values
   will disagree by about that much, on that leg only.
"""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from hitlo.io import load_streams, load_both_polar_streams
from hitlo.detectors import detect
from hitlo.symmetry import (trim_peaks, compute_step_times,
                            compute_symmetry_index)


def _si_for(left, right, method, trim):
    lr = detect(left, None, method=method)
    rr = detect(right, None, method=method)
    lt = np.asarray(lr.heel_strike_times, float)
    rt = np.asarray(rr.heel_strike_times, float)
    if len(lt) < 4 or len(rt) < 4:
        return None
    t0 = min(left.timestamps[0], right.timestamps[0])
    t1 = max(left.timestamps[-1], right.timestamps[-1])
    lt = np.sort(trim_peaks(lt, t0, t1, trim))
    rt = np.sort(trim_peaks(rt, t0, t1, trim))
    rs, ls = compute_step_times(lt, rt)
    n = min(len(rs), len(ls))
    if n < 2:
        return None
    si, per = compute_symmetry_index(rs[:n], ls[:n], signed=True)
    return dict(si=si, per=per, n=n, lt=lt, rt=rt,
                l_step=float(np.mean(ls[:n])), r_step=float(np.mean(rs[:n])),
                sem=float(np.std(per, ddof=1) / np.sqrt(len(per))))


def _pair_offsets(a_times, b_times, max_ms=200.0):
    """Median signed offset (b - a) for events that pair within max_ms."""
    out = []
    for x in a_times:
        if len(b_times) == 0:
            break
        j = int(np.argmin(np.abs(b_times - x)))
        d = (b_times[j] - x) * 1000.0
        if abs(d) <= max_ms:
            out.append(d)
    return np.asarray(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare accelerometer and gyroscope heel-strike detection.")
    ap.add_argument('xdf_file')
    ap.add_argument('--trim', type=float, default=3.0)
    ap.add_argument('--plot', default=None, help='write a comparison figure')
    a = ap.parse_args()

    if not os.path.exists(a.xdf_file):
        print(f"File not found: {a.xdf_file}")
        return 1

    left, right = load_streams(a.xdf_file, {'Sensing': {'backend': 'trigno'}})
    if left is None or right is None:
        left, right = load_both_polar_streams(a.xdf_file)
    if left is None or right is None:
        print("Could not load left/right streams from this file.")
        return 1

    print("=" * 66)
    print(f"  DETECTOR COMPARISON — {Path(a.xdf_file).name}")
    print("=" * 66)
    print(f"  duration {left.timestamps[-1] - left.timestamps[0]:.0f}s   "
          f"L {left.actual_fs:.1f} Hz   R {right.actual_fs:.1f} Hz   "
          f"backend {left.backend}")

    if not (left.has_gyro and right.has_gyro):
        print("\n  This recording has no gyroscope data, so there is nothing to")
        print("  compare — the gyro detector needs it. Polar files are")
        print("  accelerometer only; record with the Trigno bridge to use this.")
        return 1

    res = {}
    for m in ('accel', 'gyro'):
        r = _si_for(left, right, m, a.trim)
        if r is None:
            print(f"\n  {m}: too few strides to analyze")
            return 1
        res[m] = r

    print(f"\n  {'method':<8} {'SI':>9} {'SEM':>7} {'strides':>8} "
          f"{'L step':>8} {'R step':>8}")
    print("  " + "-" * 56)
    for m in ('accel', 'gyro'):
        r = res[m]
        print(f"  {m:<8} {r['si']:+8.2f}% {r['sem']:6.2f} {r['n']:8d} "
              f"{r['l_step']:7.3f}s {r['r_step']:7.3f}s")

    d_si = res['gyro']['si'] - res['accel']['si']
    print(f"\n  SI difference (gyro - accel): {d_si:+.2f} points")

    # Per-leg timing offset. This is the number that explains an SI shift.
    print(f"\n  Where each method places contact (gyro - accel, ms):")
    offs = {}
    for side, key in (('LEFT', 'lt'), ('RIGHT', 'rt')):
        o = _pair_offsets(res['accel'][key], res['gyro'][key])
        offs[side] = o
        if len(o) == 0:
            print(f"    {side:5s}  no events paired within 200 ms")
            continue
        print(f"    {side:5s}  median {np.median(o):+7.1f} ms   "
              f"IQR [{np.percentile(o, 25):+.1f}, {np.percentile(o, 75):+.1f}]   "
              f"n={len(o)}")

    if len(offs.get('LEFT', [])) and len(offs.get('RIGHT', [])):
        diff = float(np.median(offs['LEFT']) - np.median(offs['RIGHT']))
        stride = res['gyro']['l_step'] + res['gyro']['r_step']
        # Shifting one leg's events by d lengthens one step by d and shortens
        # the other by d, so SI = 200*(R-L)/stride moves by -400*d/stride.
        pred = -0.4 * diff / max(stride, 1e-6)
        print(f"\n    differential (LEFT - RIGHT): {diff:+.1f} ms")
        print(f"    A shared offset cancels out of the symmetry index; only")
        print(f"    this differential reaches it. At a {stride:.2f}s stride the")
        print(f"    sensitivity is {0.4 / max(stride, 1e-6):.2f} SI points per ms,")
        print(f"    so this predicts {pred:+.1f} points -- observed {d_si:+.2f}.")
        if abs(pred) > 1 and abs(d_si - pred) > 0.5 * abs(pred):
            print(f"    These disagree by more than half, so the offset is not")
            print(f"    the whole story: the two methods are also disagreeing")
            print(f"    about WHICH strides exist, not just when they occur.")

    if a.plot:
        _figure(left, right, res, a.plot)
        print(f"\n  figure: {a.plot}")

    print("\n  Note: these are different instants, not competing estimates of")
    print("  the same one. Neither is 'wrong'. What matters is whether the")
    print("  difference is the same on both legs — if it is, the choice of")
    print("  detector does not move your symmetry index.")
    return 0


def _figure(left, right, res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, SURF, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#fcfcfb", "#e4e3de"
    plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF,
                         'savefig.facecolor': SURF, 'axes.edgecolor': GRID,
                         'text.color': INK, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 150})
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    ax = axes[0]
    for m, c in (('accel', BLUE), ('gyro', ORANGE)):
        ax.plot(res[m]['per'], color=c, lw=1.6, marker='o', ms=4, label=m)
    ax.axhline(0, color=INK, lw=1, ls=':')
    ax.set_ylabel('per-stride SI (%)')
    ax.legend(frameon=False)
    ax.set_title('Per-stride symmetry index, both detectors on the same walk',
                 loc='left', fontweight='bold')
    ax.grid(True, alpha=.5); ax.set_axisbelow(True)
    ax = axes[1]
    for side, key, c in (('left', 'lt', BLUE), ('right', 'rt', ORANGE)):
        o = _pair_offsets(res['accel'][key], res['gyro'][key])
        if len(o):
            ax.hist(o, bins=24, alpha=.6, color=c, label=f'{side} (n={len(o)})')
    ax.axvline(0, color=INK, lw=1, ls=':')
    ax.set_xlabel('gyro contact minus accel contact (ms)')
    ax.set_ylabel('strides')
    ax.legend(frameon=False)
    ax.set_title('Timing offset between methods, per leg',
                 loc='left', fontweight='bold')
    ax.grid(True, alpha=.5); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
