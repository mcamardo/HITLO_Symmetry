"""
apps/explain_pipeline.py — Visualize every stage of the heel-strike
detection pipeline as a 7-panel matplotlib figure.

Companion to apps/diagnose_trial.py: where diagnose shows the FINAL output
(accepted/rejected on raw + jerk traces), this script walks through each
TRANSFORMATION in the pipeline so you can see what each step does to the
signal. Useful for documentation, prelim figures, lab meetings.

Pipeline stages shown (top to bottom):
    1. Raw three-axis acceleration
    2. Magnitude  |a|  (orientation invariant)
    3. Raw jerk  |d|a|/dt|
    4. Lowpass 15 Hz Butterworth (filtfilt) + z-score
    5. Two-pass peak detection (strict + recovery)
    6. Cluster grouping (pink = multi-peak, green = singleton)
    7. Cluster-keep-last + stance check (final accepted vs rejected)

Usage
-----
    cd ~/HITLO_Symmetry
    python apps/explain_pipeline.py /path/to/trial.xdf

    # Specify the 6-second window (relative to trial start)
    python apps/explain_pipeline.py trial.xdf --start 12 --end 18

    # Auto-pick window centered on a multi-peak cluster (default)
    # Custom output path
    python apps/explain_pipeline.py trial.xdf --save-png ~/figs/p048_pipeline.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Make the hitlo package importable when running from the repo root or apps/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hitlo.detection import DetectionConfig, detect_heelstrikes_full
from hitlo.io import load_both_polar_streams


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('xdf_file', help='Path to the XDF trial file')
    p.add_argument('--start', type=float, default=None,
                   help='Window start in seconds (relative to trial start). '
                        'Default: auto-pick around first multi-peak cluster.')
    p.add_argument('--end', type=float, default=None,
                   help='Window end in seconds. Default: --start + 6.')
    p.add_argument('--save-png', default='/tmp/pipeline_explainer.png',
                   help='Output PNG path.')
    p.add_argument('--dpi', type=int, default=140,
                   help='Output DPI (use 300+ for print).')
    p.add_argument('--no-show', action='store_true',
                   help='Save PNG but do not open the window.')
    p.add_argument('--mg-units', action='store_true',
                   help='Plot raw mg values instead of converting to g.')
    return p.parse_args()


# ===========================================================================
# Helpers
# ===========================================================================

def find_demo_window(left_res, t_left_rel, default_dur=6.0, trim_s=3.0):
    """Auto-pick a window centered on the first multi-peak cluster."""
    trial_dur = float(t_left_rel[-1])
    for (cstart, cend) in left_res.cluster_info:
        if cend > cstart and cstart < len(t_left_rel):
            t_center = float((t_left_rel[cstart] + t_left_rel[cend]) / 2)
            if trim_s < t_center < trial_dur - trim_s:
                lo = max(trim_s, t_center - default_dur / 2)
                hi = min(trial_dur - trim_s, lo + default_dur)
                return lo, hi
    # Fallback: first steady-state window
    return trim_s, min(trim_s + default_dur, trial_dur - trim_s)


def slice_to_window(stream, result, raw_jerk_full, jerk_sm_full,
                    display_origin: float, window_dur: float):
    """Slice all signals + indices to [display_origin, display_origin + window_dur]."""
    rel_t = stream.timestamps - display_origin
    mask = (rel_t >= 0) & (rel_t <= window_dur)
    where = np.where(mask)[0]
    if len(where) == 0:
        raise ValueError(f"No samples for {stream.name} in the chosen window")
    i0, i1 = int(where[0]), int(where[-1]) + 1

    def in_win(idx_arr):
        idx = np.asarray(idx_arr, dtype=int)
        m = (idx >= i0) & (idx < i1)
        return (idx[m] - i0).astype(int)

    return {
        'accel':         stream.accel[i0:i1],
        'times':         stream.timestamps[i0:i1] - display_origin,
        'magnitude':     result.magnitude[i0:i1],
        'raw_jerk':      raw_jerk_full[i0:i1],
        'jerk_smoothed': jerk_sm_full[i0:i1],
        'jerk_z':        result.jerk_z[i0:i1],
        'strict':        in_win(result.strict_peaks),
        'recovered':     in_win(result.recovered_peaks),
        'all_cand':      in_win(result.all_candidates),
        'accepted':      in_win(result.heel_strike_indices),
        'rejected':      in_win(result.rejected_peaks),
        'clusters': [(int(c0 - i0), int(c1 - i0))
                     for (c0, c1) in result.cluster_info
                     if c0 >= i0 and c1 < i1],
    }


def shade_clusters(ax, t, clusters):
    for (c0, c1) in clusters:
        if c0 >= len(t) or c1 >= len(t):
            continue
        if c0 == c1:
            ax.axvspan(t[c0] - 0.04, t[c1] + 0.04,
                       color='limegreen', alpha=0.10, zorder=0)
        else:
            ax.axvspan(t[c0], t[c1], color='salmon', alpha=0.20, zorder=0)


# ===========================================================================
# Plot
# ===========================================================================

def make_figure(W, baseline, cfg, scale, unit_label, suptitle, save_path,
                dpi=140, show=True):
    """Build the 7-panel figure. `W` is the windowed-signals dict from slice_to_window."""
    t = W['times']
    ax_, ay_, az_ = (W['accel'][:, 0] / scale,
                     W['accel'][:, 1] / scale,
                     W['accel'][:, 2] / scale)
    mag = W['magnitude'] / scale
    raw_jerk = W['raw_jerk'] / scale       # (g or mg) / s
    jerk_z = W['jerk_z']                   # always dimensionless

    fig, axes = plt.subplots(7, 1, figsize=(13, 16), sharex=True)
    plt.subplots_adjust(hspace=0.35)

    # ---- 1. Raw 3-axis ----
    ax = axes[0]
    ax.plot(t, ax_, color='#c0392b', lw=0.8, alpha=0.85, label='ax')
    ax.plot(t, ay_, color='#27ae60', lw=0.8, alpha=0.85, label='ay')
    ax.plot(t, az_, color='#2980b9', lw=0.8, alpha=0.85, label='az')
    ax.set_title('1. Raw three-axis acceleration  '
                 '(left shank, Polar H10 @ 200 Hz nominal)',
                 loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel(unit_label)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)

    # ---- 2. Magnitude ----
    ax = axes[1]
    ax.plot(t, mag, color='steelblue', lw=0.8)
    ax.axhline(baseline, color='gray', ls='-.', lw=0.7, alpha=0.6,
               label=f'baseline (median |a| = {baseline:.2f} {unit_label})')
    ax.set_title(r'2. Magnitude $|a| = \sqrt{x^2 + y^2 + z^2}$  '
                 r'(orientation invariant; stance ≈ 1 g)',
                 loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'|a|  ({unit_label})')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)

    # ---- 3. Raw jerk ----
    ax = axes[2]
    ax.plot(t, raw_jerk, color='steelblue', lw=0.6, alpha=0.8)
    ax.set_title(r'3. Raw jerk  $|d|a|/dt|$  '
                 r'(emphasizes sudden transitions; suppresses slow drift)',
                 loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'jerk  ({unit_label}/s)')
    ax.grid(alpha=0.2)

    # ---- 4. Lowpass + z-score ----
    ax = axes[3]
    ax.plot(t, jerk_z, color='steelblue', lw=0.8)
    ax.axhline(cfg.strict_thresh, color='#e67e22', ls='--', lw=1.0,
               label=f"strict thresh ({cfg.strict_thresh} SD)")
    ax.axhline(cfg.recovery_thresh, color='#c0392b', ls=':', lw=1.0,
               label=f"recovery thresh ({cfg.recovery_thresh} SD)")
    ax.axhline(0, color='gray', lw=0.4, alpha=0.5)
    ax.set_title(f"4. Lowpass {cfg.smooth_cutoff_hz:.0f} Hz Butterworth (filtfilt) "
                 f"+ z-score  (thresholds now in SD units)",
                 loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel('z-score')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)

    # ---- 5. Two-pass peak detection ----
    ax = axes[4]
    ax.plot(t, jerk_z, color='steelblue', lw=0.6, alpha=0.7)
    ax.axhline(cfg.strict_thresh, color='#e67e22', ls='--', lw=1.0, alpha=0.6)
    ax.axhline(0, color='gray', lw=0.4, alpha=0.5)
    if len(W['strict']):
        safe = W['strict'][W['strict'] < len(jerk_z)]
        ax.plot(t[safe], jerk_z[safe], 'o', color='navy', ms=7, zorder=5,
                label=f'strict peaks ({len(W["strict"])})')
    if len(W['recovered']):
        safe = W['recovered'][W['recovered'] < len(jerk_z)]
        ax.plot(t[safe], jerk_z[safe], 's', color='#8e44ad', ms=7, zorder=5,
                label=f'recovered peaks ({len(W["recovered"])})')
    ax.set_title(f"5. Two-pass peak detection  "
                 f"(Pass 1: ≥ {cfg.strict_thresh} SD with "
                 f"{int(cfg.min_peak_dist_s*1000)} ms min separation;  "
                 f"Pass 2: gaps > {cfg.gap_multiplier}× median, "
                 f"≥ {cfg.recovery_thresh} SD)",
                 loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel('z-score')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)

    # ---- 6. Cluster grouping ----
    ax = axes[5]
    shade_clusters(ax, t, W['clusters'])
    ax.plot(t, jerk_z, color='steelblue', lw=0.6, alpha=0.7)
    ax.axhline(0, color='gray', lw=0.4, alpha=0.5)
    if len(W['all_cand']):
        safe = W['all_cand'][W['all_cand'] < len(jerk_z)]
        ax.plot(t[safe], jerk_z[safe], 'o', color='gray', ms=6, zorder=4,
                mfc='none', mew=1.2,
                label=f'all candidates ({len(W["all_cand"])})')
    ax.set_title(f"6. Cluster grouping  "
                 f"(peaks within {int(cfg.cluster_gap_s*1000)} ms = same cluster; "
                 f"pink = multi-peak, green = singleton)",
                 loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel('z-score')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)

    # ---- 7. Cluster-keep-last + stance check ----
    ax = axes[6]
    shade_clusters(ax, t, W['clusters'])
    ax.plot(t, mag, color='steelblue', lw=0.6, alpha=0.8)
    ax.axhline(baseline, color='gray', ls='-.', lw=0.7, alpha=0.5,
               label=f'baseline ({baseline:.2f} {unit_label})')
    if len(W['accepted']):
        safe = W['accepted'][W['accepted'] < len(mag)]
        ax.plot(t[safe], mag[safe], 'v', color='navy', ms=10, zorder=6,
                label=f'accepted heel strikes ({len(W["accepted"])})')
    if len(W['rejected']):
        safe = W['rejected'][W['rejected'] < len(mag)]
        ax.plot(t[safe], mag[safe], 'x', color='gray', ms=10, mew=2.0, zorder=5,
                label=f'rejected ({len(W["rejected"])})')
    ax.set_title("7. Cluster-keep-last + stance check  "
                 "(within each cluster, scan from last peak backwards; pick "
                 "first peak above baseline AND followed by ~200 ms of stance)",
                 loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'|a|  ({unit_label})')
    ax.set_xlabel('time within window  (s)')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)

    fig.suptitle(suptitle, fontsize=12, y=0.995)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {save_path}")
    if show:
        plt.show()


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    print(f"Loading {args.xdf_file} ...")
    left, right = load_both_polar_streams(args.xdf_file)
    if left is None:
        print("ERROR: missing 'polar accel left' stream.")
        return 1
    print(f"  Left:  {len(left.accel)} samples ({left.actual_fs:.2f} Hz)")
    if right is not None:
        print(f"  Right: {len(right.accel)} samples ({right.actual_fs:.2f} Hz)")

    cfg = DetectionConfig()
    print("Running detection on left shank (full trial)...")
    L_res = detect_heelstrikes_full(left.accel, left.timestamps, cfg=cfg)
    n_multi = sum(1 for c in L_res.cluster_info if c[1] > c[0])
    print(f"  LEFT: {len(L_res.heel_strike_indices)} accepted, "
          f"{len(L_res.rejected_peaks)} rejected, "
          f"{n_multi} multi-peak clusters")

    # Re-derive raw + smoothed jerk (DetectionResult only stores jerk_z)
    raw_jerk = np.abs(np.diff(L_res.magnitude) * cfg.fs)
    raw_jerk = np.concatenate([[0.0], raw_jerk])
    b, a = butter(4, cfg.smooth_cutoff_hz / (0.5 * cfg.fs), btype='low')
    jerk_sm = filtfilt(b, a, raw_jerk)

    # Pick window
    trial_t0 = float(left.timestamps[0])
    if right is not None:
        trial_t0 = min(trial_t0, float(right.timestamps[0]))
    t_left_rel = left.timestamps - trial_t0
    if args.start is None:
        t_lo, t_hi = find_demo_window(L_res, t_left_rel)
        print(f"Auto-picked window: {t_lo:.2f}s – {t_hi:.2f}s "
              f"(centered on first multi-peak cluster)")
    else:
        t_lo = args.start
        t_hi = args.end if args.end is not None else (args.start + 6.0)
        print(f"Window: {t_lo:.2f}s – {t_hi:.2f}s")

    display_origin = trial_t0 + t_lo
    window_dur = t_hi - t_lo

    W = slice_to_window(left, L_res, raw_jerk, jerk_sm,
                        display_origin, window_dur)
    print(f"  Window contains {len(W['accepted'])} accepted heel strikes, "
          f"{len(W['rejected'])} rejected, "
          f"{sum(1 for c in W['clusters'] if c[1]>c[0])} multi-peak clusters")

    # Unit handling: Polar H10 streams in mg; convert to g unless --mg-units
    if args.mg_units:
        scale = 1.0
        unit_label = 'mg'
    else:
        scale = 1000.0
        unit_label = 'g'

    baseline = float(np.median(L_res.magnitude)) / scale

    suptitle = (f"HITLO_Symmetry detection pipeline — left shank\n"
                f"{Path(args.xdf_file).stem}  ·  "
                f"window {t_lo:.1f}–{t_hi:.1f} s  ·  "
                f"{len(W['accepted'])} heel strikes accepted, "
                f"{len(W['rejected'])} rejected")

    make_figure(W, baseline, cfg, scale, unit_label, suptitle,
                save_path=args.save_png, dpi=args.dpi,
                show=not args.no_show)
    return 0


if __name__ == '__main__':
    sys.exit(main())