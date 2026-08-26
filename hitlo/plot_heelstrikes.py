#!/usr/bin/env python3
"""
plot_heelstrikes.py — standalone heel-strike QC plot from a saved XDF.

Uses the SAME detection / IO / symmetry pipeline as the BO cost
(hitlo.detection, hitlo.io, hitlo.symmetry), so what you see here is what
got scored. No Streamlit — just matplotlib, run from the terminal.

Usage:
    python plot_heelstrikes.py /path/to/sub-P062_ses-S001_task-Pre_run-001_eeg.xdf
    python plot_heelstrikes.py <file.xdf> --trim 3.0 --save qc.png

Run from the repo root (so `hitlo` is importable), or pass --repo /path/to/repo.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _import_hitlo(repo_root: Path):
    """Make the hitlo package importable, then import what we need."""
    sys.path.insert(0, str(repo_root))
    try:
        from hitlo.detection import detect_heelstrikes_full, DetectionConfig
        from hitlo.io import load_both_polar_streams
        from hitlo.symmetry import (
            compute_step_times, compute_symmetry_index, trim_peaks,
        )
    except ModuleNotFoundError as e:
        sys.exit(
            f"Could not import the hitlo package ({e}).\n"
            f"Run this from your repo root, or pass --repo /path/to/HITLO_Symmetry."
        )
    return (detect_heelstrikes_full, DetectionConfig, load_both_polar_streams,
            compute_step_times, compute_symmetry_index, trim_peaks)


def analyze(xdf_path, trim_seconds, hitlo):
    (detect_heelstrikes_full, DetectionConfig, load_both_polar_streams,
     compute_step_times, compute_symmetry_index, trim_peaks) = hitlo

    cfg = DetectionConfig()  # same defaults as the BO cost
    left, right = load_both_polar_streams(str(xdf_path))
    if left is None or right is None:
        sys.exit("Could not load XDF or one of the streams is missing.")

    left_result = detect_heelstrikes_full(left.accel, left.timestamps, cfg=cfg)
    right_result = detect_heelstrikes_full(right.accel, right.timestamps, cfg=cfg)

    trial_start = min(left.timestamps[0], right.timestamps[0])
    trial_end = max(left.timestamps[-1], right.timestamps[-1])
    l_times = trim_peaks(left_result.heel_strike_times,
                         trial_start, trial_end, trim_seconds)
    r_times = trim_peaks(right_result.heel_strike_times,
                         trial_start, trial_end, trim_seconds)
    t_lo = trial_start + trim_seconds if trim_seconds > 0 else trial_start
    t_hi = trial_end - trim_seconds if trim_seconds > 0 else trial_end

    l_cv = (np.std(np.diff(l_times)) / np.mean(np.diff(l_times))
            if len(l_times) > 1 else np.nan)
    r_cv = (np.std(np.diff(r_times)) / np.mean(np.diff(r_times))
            if len(r_times) > 1 else np.nan)

    r_steps, l_steps = compute_step_times(l_times, r_times)
    n = min(len(r_steps), len(l_steps))
    if n >= 2:
        si_signed, _ = compute_symmetry_index(r_steps[:n], l_steps[:n], signed=True)
        si_unsigned, _ = compute_symmetry_index(r_steps[:n], l_steps[:n], signed=False)
        stride = float(r_steps[:n].mean() + l_steps[:n].mean())
        l_step_mean = float(l_steps[:n].mean())
        r_step_mean = float(r_steps[:n].mean())
    else:
        si_signed = si_unsigned = stride = l_step_mean = r_step_mean = np.nan

    drift_pct = abs(left.actual_fs - right.actual_fs) / 200.0 * 100.0

    return dict(left=left, right=right,
                left_result=left_result, right_result=right_result,
                l_times=l_times, r_times=r_times, t_lo=t_lo, t_hi=t_hi,
                l_cv=l_cv, r_cv=r_cv, drift_pct=drift_pct,
                si_signed=si_signed, si_unsigned=si_unsigned, stride=stride,
                l_step_mean=l_step_mean, r_step_mean=r_step_mean, cfg=cfg)


def print_warnings(qc):
    msgs = []
    if qc['drift_pct'] > 1.0:
        msgs.append(f"[info] clock drift {qc['drift_pct']:.2f}% "
                    f"(L={qc['left'].actual_fs:.1f} Hz, R={qc['right'].actual_fs:.1f} Hz)")
    n_l, n_r = len(qc['l_times']), len(qc['r_times'])
    if n_l < 10 or n_r < 10:
        msgs.append(f"[WARN] low heel-strike count after trim: L={n_l}, R={n_r}")
    if abs(n_l - n_r) > 3:
        msgs.append(f"[WARN] heel-strike count mismatch: L={n_l} vs R={n_r}")
    if not np.isnan(qc['l_cv']) and qc['l_cv'] > 0.25:
        msgs.append(f"[WARN] left CV={qc['l_cv']:.3f} (>0.25) — erratic timing")
    if not np.isnan(qc['r_cv']) and qc['r_cv'] > 0.25:
        msgs.append(f"[WARN] right CV={qc['r_cv']:.3f} (>0.25) — erratic timing")
    if not np.isnan(qc['si_signed']) and abs(qc['si_signed']) > 40:
        msgs.append(f"[WARN] SI={qc['si_signed']:+.1f}% — unrealistically large "
                    f"(possible sensor swap or detection error)")
    if not msgs:
        print("QC clean — no warnings.")
    else:
        for m in msgs:
            print(m)


def make_plot(qc, trim_seconds, save_path=None):
    left, right = qc['left'], qc['right']
    lr, rr = qc['left_result'], qc['right_result']

    t0 = min(left.timestamps[0], right.timestamps[0])
    t_left = left.timestamps - t0
    t_right = right.timestamps - t0
    trim_lo = qc['t_lo'] - t0
    trim_hi = qc['t_hi'] - t0
    max_t = max(t_left[-1], t_right[-1])

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    def shade_trim(ax):
        if trim_seconds > 0:
            ax.axvspan(0, trim_lo, color='gray', alpha=0.18)
            ax.axvspan(trim_hi, max_t, color='gray', alpha=0.18)

    def shade_clusters(ax, clusters, ts):
        for (cstart, cend) in clusters:
            if cstart >= len(ts) or cend >= len(ts):
                continue
            if cstart == cend:
                ax.axvspan(ts[cstart] - 0.04, ts[cend] + 0.04,
                           color='limegreen', alpha=0.15)
            else:
                ax.axvspan(ts[cstart], ts[cend], color='salmon', alpha=0.22)

    # ---- LEFT magnitude ----
    ax = axes[0]
    shade_clusters(ax, lr.cluster_info, t_left)
    ax.plot(t_left, lr.magnitude, color='steelblue', lw=1.0, alpha=0.75,
            label='L magnitude')
    ax.axhline(np.median(lr.magnitude), color='gray', ls='-.', lw=1,
               label=f'baseline ({np.median(lr.magnitude):.0f})')
    if len(lr.heel_strike_indices) > 0:
        safe = lr.heel_strike_indices[lr.heel_strike_indices < len(lr.magnitude)]
        ax.plot(t_left[safe], lr.magnitude[safe], 'v', color='navy',
                ms=8, label=f'accepted ({len(lr.heel_strike_indices)})')
    if len(lr.rejected_peaks) > 0:
        safe = lr.rejected_peaks[lr.rejected_peaks < len(lr.magnitude)]
        ax.plot(t_left[safe], lr.magnitude[safe], 'x', color='gray',
                ms=7, label=f'rejected ({len(lr.rejected_peaks)})')
    shade_trim(ax)
    ax.set_ylabel('|a|')
    ax.set_title('LEFT raw magnitude  (▼ heel strike, × rejected)')
    ax.legend(loc='upper right', fontsize=8)

    # ---- RIGHT magnitude ----
    ax = axes[1]
    shade_clusters(ax, rr.cluster_info, t_right)
    ax.plot(t_right, rr.magnitude, color='tomato', lw=1.0, alpha=0.75,
            label='R magnitude')
    ax.axhline(np.median(rr.magnitude), color='gray', ls='-.', lw=1,
               label=f'baseline ({np.median(rr.magnitude):.0f})')
    if len(rr.heel_strike_indices) > 0:
        safe = rr.heel_strike_indices[rr.heel_strike_indices < len(rr.magnitude)]
        ax.plot(t_right[safe], rr.magnitude[safe], 'v', color='darkred',
                ms=8, label=f'accepted ({len(rr.heel_strike_indices)})')
    if len(rr.rejected_peaks) > 0:
        safe = rr.rejected_peaks[rr.rejected_peaks < len(rr.magnitude)]
        ax.plot(t_right[safe], rr.magnitude[safe], 'x', color='gray',
                ms=7, label=f'rejected ({len(rr.rejected_peaks)})')
    shade_trim(ax)
    ax.set_ylabel('|a|')
    ax.set_title('RIGHT raw magnitude  (▼ heel strike, × rejected)')
    ax.legend(loc='upper right', fontsize=8)

    # ---- Jerk z overlay ----
    ax = axes[2]
    ax.plot(t_left, lr.jerk_z, color='steelblue', lw=0.8, alpha=0.7, label='L jerk z')
    ax.plot(t_right, rr.jerk_z, color='tomato', lw=0.8, alpha=0.7, label='R jerk z')
    ax.axhline(qc['cfg'].strict_thresh, color='green', ls='--',
               label=f"{qc['cfg'].strict_thresh} SD strict")
    ax.axhline(qc['cfg'].recovery_thresh, color='orange', ls=':',
               label=f"{qc['cfg'].recovery_thresh} SD recovery")
    shade_trim(ax)
    ax.set_ylabel('jerk z')
    ax.set_xlabel('LSL time (s, rel. to earliest start)')
    ax.set_title('Jerk z-score overlay')
    ax.legend(loc='upper right', fontsize=8)

    # ---- Title / subtitle ----
    if not np.isnan(qc['si_signed']):
        sub = (f"SI = {qc['si_signed']:+.2f}% signed | "
               f"{qc['si_unsigned']:.2f}% unsigned | "
               f"stride = {qc['stride']:.3f}s | "
               f"L step = {qc['l_step_mean']:.3f}s, R step = {qc['r_step_mean']:.3f}s")
    else:
        sub = "Not enough step pairs to compute symmetry"
    fig.suptitle(
        f"Heel Strike QC — "
        f"L: {len(lr.all_candidates)} cand → {len(lr.heel_strike_indices)} HS | "
        f"R: {len(rr.all_candidates)} cand → {len(rr.heel_strike_indices)} HS\n{sub}",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


# No default recording: pass one explicitly. A hard-coded path here only ever
# worked on one machine, and silently plotting someone else's trial is worse
# than a clear error.
DEFAULT_REPO = str(Path(__file__).resolve().parent.parent)


def main():
    ap = argparse.ArgumentParser(description="Standalone heel-strike QC plot from an XDF.")
    ap.add_argument("xdf",
                    help="Path to the .xdf trial file")
    ap.add_argument("--trim", type=float, default=3.0,
                    help="Seconds to trim from each end (default 3.0)")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="Repo root so `hitlo` is importable")
    ap.add_argument("--save", default=None,
                    help="Save PNG to this path instead of showing interactively")
    args = ap.parse_args()

    xdf_path = Path(args.xdf).expanduser()
    if not xdf_path.exists():
        sys.exit(f"File not found: {xdf_path}")

    hitlo = _import_hitlo(Path(args.repo).resolve())
    qc = analyze(xdf_path, args.trim, hitlo)
    print_warnings(qc)
    make_plot(qc, args.trim, save_path=args.save)


if __name__ == "__main__":
    main()
