"""
apps/compare_filter_order.py — Visual comparison of OLD (diff-then-filter)
vs NEW (filter-then-diff) heel-strike detection on the same XDF.

Self-contained: implements BOTH orderings internally and uses only the
cluster + stance logic from hitlo. You don't need to modify detection.py
to run this script — it tests both orderings regardless of what's currently
in detection.py.

Output: 6-panel figure
    1-2. OLD raw |a| + OLD jerk z-score, with detected peaks
    3-4. NEW raw |a| + NEW jerk z-score, with detected peaks
    5.   ZOOM: raw |a| vs filtered |a| in one stride
    6.   ZOOM: OLD jerk_z vs NEW jerk_z overlaid, with picked-peak markers

Panel 5 is the critical one — if filter-then-diff smears the impact, you
will see the orange (filtered) line lower and wider than the black (raw).

Usage
-----
    cd ~/HITLO_Symmetry
    python3 apps/compare_filter_order.py \\
        ~/HITLO/sub-P048/ses-S001/eeg/sub-P048_ses-S001_task-Default_run-007_eeg.xdf

    # custom window / zoom
    python3 apps/compare_filter_order.py trial.xdf --start 8 --end 14 --zoom 11.4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Use hitlo for everything DOWNSTREAM of jerk computation. We compute
# jerk_z ourselves both ways, so this script works regardless of what
# order detection.py currently uses.
from hitlo.detection import (DetectionConfig, compute_magnitude,
                              cluster_keep_last)
from hitlo.io import load_both_polar_streams


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('xdf_file', help='Path to XDF trial file')
    p.add_argument('--start', type=float, default=3.0,
                   help='Window start, trial-relative seconds (default 3.0)')
    p.add_argument('--end', type=float, default=9.0,
                   help='Window end, trial-relative seconds (default 9.0)')
    p.add_argument('--zoom', type=float, default=None,
                   help='Center zoom panels on this trial-relative time. '
                        'Default: auto-pick first detected strike in window.')
    p.add_argument('--zoom-window-ms', type=float, default=400.0,
                   help='Half-width of zoom window in ms (default 400)')
    p.add_argument('--save-png', default='/tmp/filter_order_compare.png')
    p.add_argument('--no-show', action='store_true')
    return p.parse_args()


def jerk_z_OLD(accel_data, cfg):
    """OLD ordering: differentiate, then lowpass-filter the jerk."""
    mag = compute_magnitude(accel_data)
    jerk = np.abs(np.diff(mag) * cfg.fs)
    jerk = np.concatenate([[0.0], jerk])
    b, a = butter(4, cfg.smooth_cutoff_hz / (0.5 * cfg.fs), btype='low')
    jerk_sm = filtfilt(b, a, jerk)
    std = float(np.std(jerk_sm))
    if std < 1e-6:
        return np.zeros_like(jerk_sm), mag, mag.copy()
    jerk_z = (jerk_sm - np.mean(jerk_sm)) / std
    return jerk_z, mag, mag.copy()


def jerk_z_NEW(accel_data, cfg):
    """NEW ordering: lowpass-filter magnitude, then differentiate."""
    mag = compute_magnitude(accel_data)
    b, a = butter(4, cfg.smooth_cutoff_hz / (0.5 * cfg.fs), btype='low')
    mag_sm = filtfilt(b, a, mag)
    jerk_sm = np.abs(np.diff(mag_sm) * cfg.fs)
    jerk_sm = np.concatenate([[0.0], jerk_sm])
    std = float(np.std(jerk_sm))
    if std < 1e-6:
        return np.zeros_like(jerk_sm), mag, mag_sm
    jerk_z = (jerk_sm - np.mean(jerk_sm)) / std
    return jerk_z, mag, mag_sm


def detect(jerk_z, mag, cfg):
    min_dist = int(cfg.min_peak_dist_s * cfg.fs)
    strict, _ = find_peaks(jerk_z, height=cfg.strict_thresh, distance=min_dist)
    accepted, rejected, _ = cluster_keep_last(strict, mag, cfg)
    return strict, accepted, rejected


def main():
    args = parse_args()
    print(f"Loading {args.xdf_file} ...")
    left, right = load_both_polar_streams(args.xdf_file)
    if left is None:
        print("ERROR: missing 'polar accel left' stream")
        return 1
    print(f"  Left:  {len(left.accel)} samples ({left.actual_fs:.2f} Hz)")

    cfg = DetectionConfig()

    jz_old_full, mag_full, _           = jerk_z_OLD(left.accel, cfg)
    jz_new_full, _,        mag_sm_full = jerk_z_NEW(left.accel, cfg)
    strict_old, acc_old, rej_old = detect(jz_old_full, mag_full, cfg)
    strict_new, acc_new, rej_new = detect(jz_new_full, mag_full, cfg)

    print(f"\nFull trial:")
    print(f"  OLD: {len(strict_old)} strict, {len(acc_old)} accepted, "
          f"{len(rej_old)} rejected")
    print(f"  NEW: {len(strict_new)} strict, {len(acc_new)} accepted, "
          f"{len(rej_new)} rejected")

    t_full = left.timestamps - left.timestamps[0]
    mask = (t_full >= args.start) & (t_full <= args.end)
    where = np.where(mask)[0]
    if len(where) == 0:
        print(f"ERROR: no samples in window {args.start}-{args.end}s")
        return 1
    i0, i1 = int(where[0]), int(where[-1]) + 1

    def in_win(idx):
        idx = np.asarray(idx)
        m = (idx >= i0) & (idx < i1)
        return (idx[m] - i0).astype(int)

    mag    = mag_full[i0:i1]
    mag_sm = mag_sm_full[i0:i1]
    jz_old = jz_old_full[i0:i1]
    jz_new = jz_new_full[i0:i1]
    t_plot = t_full[i0:i1] - t_full[i0]
    baseline = float(np.median(mag_full))

    strict_old_w = in_win(strict_old)
    acc_old_w    = in_win(acc_old)
    strict_new_w = in_win(strict_new)
    acc_new_w    = in_win(acc_new)
    print(f"\nIn window {args.start}-{args.end}s:")
    print(f"  OLD: {len(strict_old_w)} strict, {len(acc_old_w)} accepted")
    print(f"  NEW: {len(strict_new_w)} strict, {len(acc_new_w)} accepted")

    if args.zoom is not None:
        zoom_center = args.zoom - args.start
    elif len(acc_old_w) > 0:
        zoom_center = float(t_plot[acc_old_w[0]])
    else:
        zoom_center = (t_plot[-1] - t_plot[0]) / 2
    half = args.zoom_window_ms / 1000.0
    zoom_lo, zoom_hi = zoom_center - half, zoom_center + half

    fig, axes = plt.subplots(6, 1, figsize=(15, 16))
    plt.subplots_adjust(hspace=0.55)

    # Panel 1
    ax = axes[0]
    ax.plot(t_plot, mag, color='steelblue', lw=0.7, alpha=0.85)
    ax.axhline(baseline, color='gray', ls='-.', lw=0.7, alpha=0.6,
               label=f'baseline = {baseline:.2f} g')
    if len(acc_old_w):
        ax.plot(t_plot[acc_old_w], mag[acc_old_w], 'v', color='navy', ms=10,
                zorder=5, label=f'OLD accepted ({len(acc_old_w)})')
    ax.axvspan(zoom_lo, zoom_hi, color='goldenrod', alpha=0.15,
               label='zoom region')
    ax.set_title('OLD pipeline (diff-then-filter)  -  raw |a|',
                 loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel('|a|  (g)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)

    # Panel 2
    ax = axes[1]
    ax.plot(t_plot, jz_old, color='steelblue', lw=0.7)
    ax.axhline(cfg.strict_thresh, color='#e67e22', ls='--', lw=1.0,
               label=f'strict ({cfg.strict_thresh} SD)')
    ax.axhline(0, color='gray', lw=0.4, alpha=0.5)
    if len(strict_old_w):
        ax.plot(t_plot[strict_old_w], jz_old[strict_old_w], 'o', color='navy',
                ms=6, zorder=4, label=f'strict ({len(strict_old_w)})')
    ax.axvspan(zoom_lo, zoom_hi, color='goldenrod', alpha=0.15)
    ax.set_title('OLD pipeline  -  jerk z-score',
                 loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel('z-score')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)

    # Panel 3
    ax = axes[2]
    ax.plot(t_plot, mag, color='#d35400', lw=0.7, alpha=0.85)
    ax.axhline(baseline, color='gray', ls='-.', lw=0.7, alpha=0.6,
               label=f'baseline = {baseline:.2f} g')
    if len(acc_new_w):
        ax.plot(t_plot[acc_new_w], mag[acc_new_w], 'v', color='#922b21', ms=10,
                zorder=5, label=f'NEW accepted ({len(acc_new_w)})')
    ax.axvspan(zoom_lo, zoom_hi, color='goldenrod', alpha=0.15)
    ax.set_title('NEW pipeline (filter-then-diff)  -  raw |a|',
                 loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel('|a|  (g)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)

    # Panel 4
    ax = axes[3]
    ax.plot(t_plot, jz_new, color='#d35400', lw=0.7)
    ax.axhline(cfg.strict_thresh, color='#e67e22', ls='--', lw=1.0,
               label=f'strict ({cfg.strict_thresh} SD)')
    ax.axhline(0, color='gray', lw=0.4, alpha=0.5)
    if len(strict_new_w):
        ax.plot(t_plot[strict_new_w], jz_new[strict_new_w], 'o',
                color='#922b21', ms=6, zorder=4,
                label=f'strict ({len(strict_new_w)})')
    ax.axvspan(zoom_lo, zoom_hi, color='goldenrod', alpha=0.15)
    ax.set_title('NEW pipeline  -  jerk z-score',
                 loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel('z-score')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)

    # Panel 5: zoom raw vs filtered magnitude
    ax = axes[4]
    zmask = (t_plot >= zoom_lo) & (t_plot <= zoom_hi)
    ax.plot(t_plot[zmask], mag[zmask], color='black', lw=1.5, alpha=0.85,
            label='raw |a|')
    ax.plot(t_plot[zmask], mag_sm[zmask], color='#d35400', lw=2.0,
            label='filtered |a| (NEW pipeline input to diff)')
    ax.axhline(baseline, color='gray', ls='-.', lw=0.7, alpha=0.5)
    for idx in acc_old_w:
        if zoom_lo <= t_plot[idx] <= zoom_hi:
            ax.axvline(t_plot[idx], color='navy', ls=':', lw=1.2, alpha=0.7)
    for idx in acc_new_w:
        if zoom_lo <= t_plot[idx] <= zoom_hi:
            ax.axvline(t_plot[idx], color='#922b21', ls=':', lw=1.2, alpha=0.7)
    ax.set_title('ZOOM  -  raw vs filtered |a|  '
                 '(does filter-then-diff smear the impact?)',
                 loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel('|a|  (g)')
    ax.set_xlim(zoom_lo, zoom_hi)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)

    # Panel 6: zoom both jerk_z overlaid
    ax = axes[5]
    ax.plot(t_plot[zmask], jz_old[zmask], color='steelblue', lw=2.0,
            label='OLD jerk_z (diff-then-filter)')
    ax.plot(t_plot[zmask], jz_new[zmask], color='#d35400', lw=2.0,
            label='NEW jerk_z (filter-then-diff)')
    ax.axhline(cfg.strict_thresh, color='#e67e22', ls='--', lw=1.0, alpha=0.7,
               label=f'strict ({cfg.strict_thresh} SD)')
    ax.axhline(0, color='gray', lw=0.4, alpha=0.5)
    first_old, first_new = True, True
    for idx in acc_old_w:
        if zoom_lo <= t_plot[idx] <= zoom_hi:
            ax.axvline(t_plot[idx], color='navy', ls=':', lw=1.2, alpha=0.7,
                       label='OLD picked' if first_old else None)
            first_old = False
    for idx in acc_new_w:
        if zoom_lo <= t_plot[idx] <= zoom_hi:
            ax.axvline(t_plot[idx], color='#922b21', ls=':', lw=1.2, alpha=0.7,
                       label='NEW picked' if first_new else None)
            first_new = False
    ax.set_title('ZOOM  -  jerk z-score overlay  '
                 '(timing offset = phase shift between pipelines)',
                 loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel('z-score')
    ax.set_xlabel('time within window  (s)')
    ax.set_xlim(zoom_lo, zoom_hi)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)

    fig.suptitle(f'Filter-order comparison  -  {Path(args.xdf_file).stem}\n'
                 f'OLD: {len(acc_old_w)} accepted  /  '
                 f'NEW: {len(acc_new_w)} accepted   '
                 f'({cfg.smooth_cutoff_hz:.0f} Hz Butterworth, filtfilt)',
                 fontsize=12, y=0.995)
    plt.savefig(args.save_png, dpi=140, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {args.save_png}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == '__main__':
    sys.exit(main())
