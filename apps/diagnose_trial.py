"""
apps/diagnose_trial.py — standalone trial quality diagnostic.

Takes one XDF file and produces a 4-panel matplotlib figure showing:
  1. LEFT raw magnitude with heel-strike markers and cluster shading
  2. LEFT jerk z-score
  3. RIGHT raw magnitude with heel-strike markers and cluster shading
  4. RIGHT jerk z-score

Also prints a summary report with symmetry index, sample-rate drift, warnings.

Usage
-----
    # Default (edit XDF_FILE below)
    python apps/diagnose_trial.py

    # Specify a file
    python apps/diagnose_trial.py /path/to/trial.xdf

    # With custom trim
    python apps/diagnose_trial.py /path/to/trial.xdf --trim 3.0

All detection logic lives in hitlo/ — this script is just the diagnostic
visualization layer on top.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Make the hitlo package importable when running from the repo root or apps/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hitlo.detection import (
    DetectionConfig, compute_magnitude, detect_heelstrikes_full,
)
from hitlo.symmetry import (
    compute_step_times, compute_symmetry_index,
    trim_peaks, filter_implausible_strides,
)
from hitlo.io import load_both_polar_streams


# ===========================================================================
# CLI arguments
# ===========================================================================

DEFAULT_XDF_FILE = (
    '/Users/maccamardo/HITLO/sub-P048/ses-S001/eeg/'
    'sub-P048_ses-S001_task-Default_run-007_eeg.xdf'
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose a single HITLO trial XDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('xdf_file', nargs='?', default=DEFAULT_XDF_FILE,
                   help='Path to XDF file')
    p.add_argument('--trim', type=float, default=3.0,
                   help='Trim N seconds from each end (0 disables)')
    p.add_argument('--save-png', default='/tmp/symmetry_diagnose.png',
                   help='Where to save the diagnostic figure')
    p.add_argument('--signed', action='store_true', default=True,
                   help='Use signed symmetry index (default)')
    p.add_argument('--no-show', action='store_true',
                   help='Save PNG but do not open the window')

    # Detection tuning (all optional, defaults from DetectionConfig)
    p.add_argument('--strict-thresh', type=float,
                   help='Jerk z-score threshold for strict-pass peaks')
    p.add_argument('--cluster-gap', type=float,
                   help='Cluster grouping gap in seconds')
    p.add_argument('--stance-tol', type=float,
                   help='Stance-check tolerance (fraction of baseline)')

    return p.parse_args()


# ===========================================================================
# Sanity-check helpers (diagnostic-only — not part of the core library)
# ===========================================================================

def collect_warnings(left_result, right_result,
                     left_stream, right_stream,
                     left_times, right_times,
                     trial_duration_s) -> list:
    """Run optional sanity checks and return a list of warning strings."""
    w = []

    # Sample rate drift
    drift_pct = 100 * abs(left_stream.actual_fs - right_stream.actual_fs) / \
                min(left_stream.actual_fs, right_stream.actual_fs)
    if drift_pct > 10.0:
        w.append(f"Sample rate drift {drift_pct:.1f}% between sensors "
                 f"(L={left_stream.actual_fs:.1f}Hz, "
                 f"R={right_stream.actual_fs:.1f}Hz)")

    # Heel-strike counts
    for label, res in [('LEFT', left_result), ('RIGHT', right_result)]:
        n = len(res.heel_strike_indices)
        if n < 5:
            w.append(f"[{label}] only {n} heel strikes detected (<5)")
        elif n > 100:
            w.append(f"[{label}] {n} heel strikes detected (>100) — over-detection?")
        if trial_duration_s > 5:
            rate = n / trial_duration_s
            if rate < 0.3:
                w.append(f"[{label}] heel strike rate {rate:.2f}/s is very slow")
            if rate > 3.0:
                w.append(f"[{label}] heel strike rate {rate:.2f}/s is very fast")

    # L/R balance
    n_l, n_r = len(left_times), len(right_times)
    if n_l == 0 or n_r == 0:
        w.append(f"L/R imbalance post-trim: L={n_l}, R={n_r}")
    else:
        ratio = max(n_l, n_r) / min(n_l, n_r)
        if ratio > 1.5:
            w.append(f"L/R post-trim count imbalance {n_l}:{n_r} (ratio {ratio:.2f}x)")

    return w


# ===========================================================================
# Plot
# ===========================================================================

def plot_diagnostic(left_stream, right_stream,
                    left_result, right_result,
                    trim_lo: float, trim_hi: float,
                    cfg: DetectionConfig,
                    save_path: str,
                    show: bool = True) -> None:
    """4-panel diagnostic figure.

    Panels (shared X axis, LSL seconds relative to trial start):
      1. Left raw magnitude
      2. Left jerk z-score
      3. Right raw magnitude
      4. Right jerk z-score

    On the magnitude panels, accepted heel strikes show as colored triangles,
    rejected candidates as gray X. Cluster windows shaded (pink = multi-peak,
    green = singleton) behind the signal.
    """
    plt.close('all')
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

    t0 = min(left_stream.timestamps[0], right_stream.timestamps[0])
    t_left = left_stream.timestamps - t0
    t_right = right_stream.timestamps - t0
    trim_lo_rel = trim_lo - t0
    trim_hi_rel = trim_hi - t0
    max_t = max(t_left[-1], t_right[-1])

    def shade_clusters(ax, ts, clusters):
        for (cstart, cend) in clusters:
            if cstart >= len(ts) or cend >= len(ts):
                continue
            x0, x1 = ts[cstart], ts[cend]
            if cstart == cend:
                ax.axvspan(x0 - 0.04, x1 + 0.04, color='limegreen',
                           alpha=0.12, zorder=0)
            else:
                ax.axvspan(x0, x1, color='salmon', alpha=0.22, zorder=0)

    def plot_panel(ax, t, sig, accepted, rejected, title,
                   line_color, accept_color,
                   threshold_line=None, show_baseline=False, clusters=None):
        if clusters is not None:
            shade_clusters(ax, t, clusters)
        ax.plot(t, sig, color=line_color, lw=0.6, alpha=0.8)

        if show_baseline:
            baseline = float(np.median(sig))
            ax.axhline(baseline, color='gray', ls='-.', lw=0.7, alpha=0.4,
                       label=f'baseline ({baseline:.0f})')

        if len(accepted) > 0:
            safe = accepted[accepted < len(sig)]
            ax.plot(t[safe], sig[safe], 'v', color=accept_color, ms=7, zorder=6,
                    label=f'accepted ({len(accepted)})')
        if len(rejected) > 0:
            safe = rejected[rejected < len(sig)]
            ax.plot(t[safe], sig[safe], 'x', color='gray', ms=9, zorder=5,
                    mew=1.5, label=f'rejected ({len(rejected)})')

        if trim_lo_rel > t[0]:
            ax.axvspan(t[0], trim_lo_rel, color='gray', alpha=0.15, zorder=0)
        if trim_hi_rel < max_t:
            ax.axvspan(trim_hi_rel, max_t, color='gray', alpha=0.15, zorder=0,
                       label=f'trimmed')

        if threshold_line is not None:
            ax.axhline(threshold_line[0], color='green', ls='--', lw=0.9, alpha=0.5,
                       label=f'{threshold_line[0]} SD strict')
            ax.axhline(threshold_line[1], color='orange', ls=':', lw=0.9, alpha=0.5,
                       label=f'{threshold_line[1]} SD recovery')

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Left panels
    plot_panel(axes[0], t_left, left_result.magnitude,
               left_result.heel_strike_indices, left_result.rejected_peaks,
               'LEFT — RAW MAGNITUDE (triangles = heel strikes; X = in-cluster rejected)',
               'steelblue', 'navy',
               show_baseline=True, clusters=left_result.cluster_info)
    axes[0].set_ylabel('|a|')

    plot_panel(axes[1], t_left, left_result.jerk_z,
               left_result.heel_strike_indices, left_result.rejected_peaks,
               'LEFT — JERK (pink = multi-peak cluster, green = singleton)',
               'steelblue', 'navy',
               threshold_line=(cfg.strict_thresh, cfg.recovery_thresh),
               clusters=left_result.cluster_info)
    axes[1].set_ylabel('z-score')

    # Right panels
    plot_panel(axes[2], t_right, right_result.magnitude,
               right_result.heel_strike_indices, right_result.rejected_peaks,
               'RIGHT — RAW MAGNITUDE (triangles = heel strikes; X = in-cluster rejected)',
               'tomato', 'darkred',
               show_baseline=True, clusters=right_result.cluster_info)
    axes[2].set_ylabel('|a|')

    plot_panel(axes[3], t_right, right_result.jerk_z,
               right_result.heel_strike_indices, right_result.rejected_peaks,
               'RIGHT — JERK (pink = multi-peak cluster, green = singleton)',
               'tomato', 'darkred',
               threshold_line=(cfg.strict_thresh, cfg.recovery_thresh),
               clusters=right_result.cluster_info)
    axes[3].set_ylabel('z-score')

    axes[-1].set_xlabel('LSL time (s, relative to earliest stream start)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    if show:
        plt.show()


# ===========================================================================
# Report
# ===========================================================================

def print_report(left_result, right_result,
                 left_stream, right_stream,
                 left_times, right_times,
                 right_steps, left_steps,
                 si_signed, si_unsigned, per_stride,
                 trim_s: float, signed: bool, warnings: list) -> None:
    print("\n" + "=" * 70)
    print("HITLO_Symmetry DIAGNOSTIC — cluster-keep-last heel-strike detection")
    print("=" * 70)

    print("\nSENSOR TIMING:")
    print(f"   LEFT  actual rate: {left_stream.actual_fs:.2f} Hz")
    print(f"   RIGHT actual rate: {right_stream.actual_fs:.2f} Hz")
    offset_ms = (right_stream.timestamps[0] - left_stream.timestamps[0]) * 1000
    print(f"   Start offset: {offset_ms:+.1f} ms  (right - left)")

    print("\nDETECTION PIPELINE:")
    for label, res in [('LEFT ', left_result), ('RIGHT', right_result)]:
        print(f"   {label} candidates: {len(res.all_candidates):>3}  "
              f"-> heel strikes: {len(res.heel_strike_indices):>3}  "
              f"(rejected: {len(res.rejected_peaks):>3})")

    if trim_s > 0:
        n_l_raw = len(left_result.heel_strike_indices)
        n_r_raw = len(right_result.heel_strike_indices)
        print(f"\nSTEADY-STATE TRIM:  {trim_s}s from each end")
        print(f"   LEFT  peaks after trim: {len(left_times)}  "
              f"(-{n_l_raw - len(left_times)})")
        print(f"   RIGHT peaks after trim: {len(right_times)}  "
              f"(-{n_r_raw - len(right_times)})")

    print("\nSTEP TIMES:")
    print(f"   Right steps (L->R): {len(right_steps)},  "
          f"mean = {right_steps.mean():.3f}s  (std = {right_steps.std():.3f}s)")
    print(f"   Left  steps (R->L): {len(left_steps)},  "
          f"mean = {left_steps.mean():.3f}s  (std = {left_steps.std():.3f}s)")
    print(f"   Stride time:        {right_steps.mean() + left_steps.mean():.3f}s")

    print("\nSYMMETRY INDEX:")
    print(f"   Signed mean:     {si_signed:+.3f}%")
    print(f"   Unsigned mean:   {si_unsigned:.3f}%")
    print(f"   Per-stride range: [{per_stride.min():+.2f}%, {per_stride.max():+.2f}%]")
    print(f"   Per-stride std:   {per_stride.std():.3f}%")

    print("\nINTERPRETATION:")
    a = abs(si_signed)
    if a < 2:
        tag = "Near-zero symmetry — very balanced gait"
    elif a < 10:
        tag = "Mild asymmetry"
    elif a < 20:
        tag = "Moderate asymmetry"
    else:
        tag = "Severe asymmetry"
    direction = ("Right step > left step" if si_signed > 0
                 else "Left step > right step" if si_signed < 0
                 else "")
    print(f"   {tag}")
    if direction:
        print(f"   {direction}")

    print("=" * 70)

    if warnings:
        print(f"\nSANITY-CHECK WARNINGS ({len(warnings)}):")
        print("-" * 70)
        for w in warnings:
            print(f"  ⚠  {w}")
        print("-" * 70)
    else:
        print("\n✓  All sanity checks passed.")


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    args = parse_args()

    # Build detection config (apply CLI overrides on top of defaults)
    cfg_kwargs = {}
    if args.strict_thresh is not None:
        cfg_kwargs['strict_thresh'] = args.strict_thresh
    if args.cluster_gap is not None:
        cfg_kwargs['cluster_gap_s'] = args.cluster_gap
    if args.stance_tol is not None:
        cfg_kwargs['stance_tolerance_pct'] = args.stance_tol
    cfg = DetectionConfig(**cfg_kwargs)

    print(f"Loading {args.xdf_file} ...")
    left, right = load_both_polar_streams(args.xdf_file)
    if left is None or right is None:
        print(f"❌ Missing 'polar accel left' or 'polar accel right' stream")
        return 1

    print(f"   Left:  {len(left.accel)} samples ({left.actual_fs:.2f} Hz)")
    print(f"   Right: {len(right.accel)} samples ({right.actual_fs:.2f} Hz)")

    # Run detection on both sensors
    left_result = detect_heelstrikes_full(left.accel, left.timestamps, cfg=cfg)
    right_result = detect_heelstrikes_full(right.accel, right.timestamps, cfg=cfg)

    print(f"\n[LEFT ] {len(left_result.all_candidates)} candidates "
          f"-> {len(left_result.heel_strike_indices)} heel strikes")
    print(f"[RIGHT] {len(right_result.all_candidates)} candidates "
          f"-> {len(right_result.heel_strike_indices)} heel strikes")

    # Trim
    trial_start = min(left.timestamps[0], right.timestamps[0])
    trial_end = max(left.timestamps[-1], right.timestamps[-1])
    left_times = trim_peaks(left_result.heel_strike_times,
                            trial_start, trial_end, args.trim)
    right_times = trim_peaks(right_result.heel_strike_times,
                             trial_start, trial_end, args.trim)
    trim_lo = trial_start + args.trim if args.trim > 0 else trial_start
    trim_hi = trial_end - args.trim if args.trim > 0 else trial_end

    # Plausibility filter
    left_times, _, _ = filter_implausible_strides(left_times)
    right_times, _, _ = filter_implausible_strides(right_times)

    if len(left_times) < 3 or len(right_times) < 3:
        print(f"\n❌ Not enough peaks after trim (L={len(left_times)}, "
              f"R={len(right_times)}) — try reducing --trim.")
        return 1

    # Symmetry
    right_steps, left_steps = compute_step_times(left_times, right_times)
    if len(right_steps) < 2 or len(left_steps) < 2:
        print("\n❌ Not enough step pairs to compute symmetry.")
        return 1

    si_signed, per_stride = compute_symmetry_index(right_steps, left_steps, signed=True)
    si_unsigned, _ = compute_symmetry_index(right_steps, left_steps, signed=False)

    # Warnings + report
    warnings = collect_warnings(left_result, right_result, left, right,
                                left_times, right_times,
                                trial_end - trial_start)
    print_report(left_result, right_result, left, right,
                 left_times, right_times, right_steps, left_steps,
                 si_signed, si_unsigned, per_stride,
                 args.trim, args.signed, warnings)

    # Plot
    plot_diagnostic(left, right, left_result, right_result,
                    trim_lo, trim_hi, cfg,
                    save_path=args.save_png, show=not args.no_show)

    return 0


if __name__ == '__main__':
    sys.exit(main())
