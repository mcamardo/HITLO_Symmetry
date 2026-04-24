"""
scripts/analyze_experiment.py — post-hoc analysis of a completed HITLO_Symmetry session.

Does 4 things in sequence:
  1. Process all XDF files in a session → master_gait_data.csv
  2. Generate the gait asymmetry timeline plot (full experiment overview)
  3. Visualize Bayesian Optimization progress iteration-by-iteration
  4. Plot all N trial torque curves in a grid

Unlike the old version of this script, this one uses hitlo/ library functions
(hitlo.detection, hitlo.symmetry, hitlo.cost) — meaning the post-hoc analysis
uses the EXACT SAME heel-strike detection that the BO used in real time.
No algorithmic drift between live and post-hoc.

Usage
-----
    python scripts/analyze_experiment.py --subject P049 --session S001

    # With custom base directory
    python scripts/analyze_experiment.py --subject P049 --session S001 \\
        --base-dir /path/to/data

    # Skip certain steps
    python scripts/analyze_experiment.py --subject P049 --session S001 \\
        --no-xdf-processing    # re-use existing master_gait_data.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Make hitlo importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hitlo.detection import DetectionConfig, detect_heelstrikes_full
from hitlo.symmetry import compute_step_times, compute_symmetry_index, trim_peaks
from hitlo.io import load_both_polar_streams, load_polar_stream
from hitlo.cost import compute_torque_curve


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-hoc analysis of a completed HITLO_Symmetry session.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--subject', required=True, help='Subject ID (e.g. P049)')
    p.add_argument('--session', default='S001', help='Session ID')
    p.add_argument('--base-dir', default='/Users/maccamardo/HITLO',
                   help='Base directory containing sub-*/ses-*/ folders')
    p.add_argument('--trim', type=float, default=3.0,
                   help='Trim N seconds from each end of each trial')
    p.add_argument('--no-xdf-processing', action='store_true',
                   help='Skip Part 1 (XDF → CSV). Reuse existing master_gait_data.csv.')
    p.add_argument('--no-timeline-plot', action='store_true',
                   help='Skip Part 2 (gait asymmetry timeline).')
    p.add_argument('--no-optimization-plots', action='store_true',
                   help='Skip Part 3 (BO iteration visualizations).')
    p.add_argument('--no-torque-plots', action='store_true',
                   help='Skip Part 4 (per-trial torque curves).')
    return p.parse_args()


# ===========================================================================
# Phase-code inference from filename
# ===========================================================================

def parse_xdf_filename(filename: str) -> dict:
    """Extract subject, session, task, run, and phase from a BIDS-style filename."""
    import re
    pattern = r'sub-(?P<subject>\w+)_ses-(?P<session>\w+)_task-(?P<task>\w+)_run-(?P<run>\d+)'
    match = re.search(pattern, filename)
    if not match:
        return None
    info = match.groupdict()
    info['run'] = int(info['run'])
    task = info['task']
    run = info['run']
    if task == 'Pre':
        info['phase_code'] = 'pre'
    elif task == 'Post':
        info['phase_code'] = 'post'
    elif task == 'Default':
        info['phase_code'] = 'exploration' if run <= 5 else 'bo_optimization'
    info['trial_num'] = run
    return info


# ===========================================================================
# PART 1 — XDF processing → CSV
# ===========================================================================

def process_xdf_files(data_dir: str, trim_s: float) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("PART 1: Processing XDF files using hitlo/ detection pipeline")
    print("=" * 70 + "\n")

    xdf_files = sorted(Path(data_dir).glob('*.xdf'))
    print(f"Found {len(xdf_files)} XDF files\n")

    cfg = DetectionConfig()
    all_rows = []

    for xdf_file in xdf_files:
        print(f"{xdf_file.name}...", end=' ')
        info = parse_xdf_filename(xdf_file.name)
        if not info:
            print("skip (filename mismatch)")
            continue

        # Try two-sensor first, fall back to single
        left, right = load_both_polar_streams(str(xdf_file))
        if left is None or right is None:
            sternum = load_polar_stream(str(xdf_file), 'polar accel')
            if sternum is None:
                print("no sensor data")
                continue
            # Single-sensor fallback — compute SI from alternating intervals
            result = detect_heelstrikes_full(sternum.accel, sternum.timestamps, cfg=cfg)
            hs = trim_peaks(result.heel_strike_times,
                            sternum.timestamps[0], sternum.timestamps[-1], trim_s)
            if len(hs) < 4:
                print("not enough heel strikes")
                continue
            intervals = np.diff(hs)
            per_stride = []
            for i in range(0, len(intervals) - 1, 2):
                a, b = intervals[i], intervals[i + 1]
                per_stride.append(2 * (a - b) / (a + b) * 100)
            per_stride = np.array(per_stride)
        else:
            # Two-sensor signed SI
            l_result = detect_heelstrikes_full(left.accel, left.timestamps, cfg=cfg)
            r_result = detect_heelstrikes_full(right.accel, right.timestamps, cfg=cfg)
            trial_start = min(left.timestamps[0], right.timestamps[0])
            trial_end = max(left.timestamps[-1], right.timestamps[-1])
            l_times = trim_peaks(l_result.heel_strike_times, trial_start, trial_end, trim_s)
            r_times = trim_peaks(r_result.heel_strike_times, trial_start, trial_end, trim_s)
            if len(l_times) < 3 or len(r_times) < 3:
                print("not enough heel strikes after trim")
                continue
            r_steps, l_steps = compute_step_times(l_times, r_times)
            n = min(len(r_steps), len(l_steps))
            if n < 2:
                print("not enough step pairs")
                continue
            _, per_stride = compute_symmetry_index(r_steps[:n], l_steps[:n], signed=True)

        if len(per_stride) == 0:
            print("no per-stride data")
            continue

        for i, s in enumerate(per_stride):
            all_rows.append({
                'phase_code':        info['phase_code'],
                'trial_num':         info['trial_num'],
                'step_number':       i + 1,
                'asymmetry_percent': float(s),
            })

        print(f"✓ ({len(per_stride)} strides, mean={np.mean(per_stride):+.1f}%)")

    df = pd.DataFrame(all_rows)
    csv_path = f'{data_dir}/master_gait_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved: {csv_path}\n")
    return df


# ===========================================================================
# PART 2 — Gait asymmetry timeline plot
# ===========================================================================

def create_timeline_plot(df: pd.DataFrame, data_dir: str) -> None:
    print("=" * 70)
    print("PART 2: Gait asymmetry timeline plot")
    print("=" * 70 + "\n")

    if len(df) == 0:
        print("⚠️  No data to plot\n")
        return

    df = df.copy()
    df['asymmetry_display'] = df['asymmetry_percent'].abs()

    # Auto-detect trial groups (works for any n_steps, not just 15)
    trial_keys = sorted(
        df.groupby(['phase_code', 'trial_num']).groups.keys(),
        key=lambda x: (
            {'pre': 0, 'exploration': 1, 'bo_optimization': 2, 'post': 3}.get(x[0], 4),
            x[1]
        )
    )

    phase_colors = {
        'pre': '#95a5a6',
        'exploration': '#e74c3c',
        'bo_optimization': '#3498db',
        'post': '#27ae60',
    }

    fig, ax = plt.subplots(figsize=(max(20, len(trial_keys) * 1.5), 7))
    y_max = df['asymmetry_display'].max() * 1.2 if len(df) > 0 else 10
    x_offset = 0

    for phase, trial in trial_keys:
        mask = (df['phase_code'] == phase) & (df['trial_num'] == trial)
        if not mask.any():
            continue
        trial_data = df.loc[mask].copy()
        n = len(trial_data)
        times = np.arange(n) + x_offset
        color = phase_colors.get(phase, 'gray')
        label = f"{phase[:4]}\n{trial}"

        ax.scatter(times, trial_data['asymmetry_display'],
                   c=color, alpha=0.5, s=15, edgecolors='none', rasterized=True)
        mean_asymm = trial_data['asymmetry_display'].mean()
        ax.text(x_offset + n / 2, y_max * 0.95,
                f"{label}\n{mean_asymm:.1f}%",
                ha='center', va='top', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                          alpha=0.4, edgecolor='none'))
        x_offset += n + 30

    ax.set_xlabel('Step Number', fontsize=15, fontweight='bold')
    ax.set_ylabel('|Gait Asymmetry| (%)', fontsize=15, fontweight='bold')
    ax.set_title('Gait Asymmetry Timeline: HIL Exoskeleton Optimization',
                 fontsize=17, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
    ax.set_ylim(bottom=0, top=y_max)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        p = f'{data_dir}/gait_asymmetry_timeline.{ext}'
        plt.savefig(p, dpi=300 if ext == 'png' else None,
                    bbox_inches='tight', facecolor='white')
        print(f"✅ {p}")
    plt.close()
    print()


# ===========================================================================
# PART 3 — BO iteration visualizations
# ===========================================================================

def load_iteration_data(iter_path: Path,
                         range_arr: np.ndarray) -> tuple:
    """Load one BO iteration's data.csv and denormalize R, L0."""
    csv_path = iter_path / 'data.csv'
    if not csv_path.exists():
        return None, None, None
    data = np.loadtxt(csv_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    R_norm = data[:, 0]
    L0_norm = data[:, 1]
    bo_cost = data[:, 2]
    R = R_norm * (range_arr[1, 0] - range_arr[0, 0]) + range_arr[0, 0]
    L0 = L0_norm * (range_arr[1, 1] - range_arr[0, 1]) + range_arr[0, 1]
    return R, L0, bo_cost


def plot_bo_iteration(iter_path: Path, iter_num: int,
                      output_dir: Path, hil_results: pd.DataFrame,
                      range_arr: np.ndarray) -> tuple:
    print(f"\nVisualizing iteration {iter_num}...")
    R, L0, bo_cost = load_iteration_data(iter_path, range_arr)
    if R is None:
        print("  ❌ No data found")
        return None, None, None

    actual_asymm = hil_results['cost'].values[:iter_num]
    n = min(len(R), len(actual_asymm))
    R, L0, bo_cost = R[:n], L0[:n], bo_cost[:n]
    actual_asymm = actual_asymm[:n]

    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(131, projection='3d')
    sc1 = ax1.scatter(R, L0, bo_cost, c=bo_cost, s=100, cmap='RdYlGn',
                      edgecolors='black', linewidths=2)
    ax1.set_title(f'BO Cost (Iter {iter_num})', fontweight='bold')
    ax1.set_xlabel('R (m)'); ax1.set_ylabel('L0 (m)'); ax1.set_zlabel('BO Score')
    fig.colorbar(sc1, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(132, projection='3d')
    sc2 = ax2.scatter(R, L0, np.abs(actual_asymm), c=np.abs(actual_asymm),
                      s=100, cmap='RdYlGn_r', edgecolors='black', linewidths=2)
    ax2.set_title(f'|Asymmetry| (Iter {iter_num})', fontweight='bold')
    ax2.set_xlabel('R (m)'); ax2.set_ylabel('L0 (m)'); ax2.set_zlabel('|SI %|')
    fig.colorbar(sc2, ax=ax2, shrink=0.5)

    ax3 = fig.add_subplot(133)
    ax3.plot(R, L0, 'o-', linewidth=2, markersize=10, color='royalblue',
             markeredgecolor='black', markeredgewidth=2)
    best_idx = np.argmin(np.abs(actual_asymm))
    ax3.scatter(R[best_idx], L0[best_idx], s=300, c='gold', marker='*',
                edgecolors='black', linewidths=2, zorder=10)
    for i, (r, l, a) in enumerate(zip(R, L0, actual_asymm)):
        ax3.annotate(f'{i+1}\n{a:.1f}%', (r, l), fontsize=8, fontweight='bold',
                     ha='center', va='center')
    ax3.set_xlabel('R (m)'); ax3.set_ylabel('L0 (m)')
    ax3.set_title('Parameter Trajectory', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f'optimization_iteration_{iter_num:02d}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ {output_path.name}")
    return R, L0, actual_asymm


def create_optimization_summary(all_data: list, output_dir: Path) -> None:
    print("\nCreating optimization summary...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    iters, best_asymm, best_R, best_L0 = [], [], [], []
    for iter_num, R, L0, asymm in all_data:
        iters.append(iter_num)
        best_idx = np.argmin(np.abs(asymm))
        best_asymm.append(asymm[best_idx])
        best_R.append(R[best_idx])
        best_L0.append(L0[best_idx])

    ax1.plot(iters, np.abs(best_asymm), 'o-', linewidth=2, markersize=8,
             color='#3498db', markeredgecolor='black', markeredgewidth=2)
    ax1.set_xlabel('Iteration', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Best |Asymmetry| (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Best |Asymmetry| per Iteration', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(best_R, best_L0, 'o-', linewidth=2, markersize=8,
             color='#e74c3c', markeredgecolor='black', markeredgewidth=2)
    for i, (r, l) in enumerate(zip(best_R, best_L0)):
        ax2.annotate(f'{iters[i]}', (r, l), fontsize=10, fontweight='bold',
                     xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('R (m)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('L0 (m)', fontweight='bold', fontsize=12)
    ax2.set_title('Best Parameters Trajectory', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = output_dir / 'optimization_summary.png'
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ {p.name}")


def plot_best_vs_worst_torque(all_data: list, output_dir: Path) -> None:
    print("\nCreating best vs worst torque comparison...")
    best_abs, worst_abs = float('inf'), -float('inf')
    best_p = worst_p = None

    for iter_num, R, L0, asymm in all_data:
        bi = np.argmin(np.abs(asymm))
        wi = np.argmax(np.abs(asymm))
        if np.abs(asymm[bi]) < best_abs:
            best_abs = np.abs(asymm[bi])
            best_p = (R[bi], L0[bi], asymm[bi], iter_num)
        if np.abs(asymm[wi]) > worst_abs:
            worst_abs = np.abs(asymm[wi])
            worst_p = (R[wi], L0[wi], asymm[wi], iter_num)

    if best_p is None or worst_p is None:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    ang_w, tor_w = compute_torque_curve(worst_p[0], worst_p[1])
    ax.plot(ang_w, tor_w, color='red', linewidth=3, linestyle='--', alpha=0.7,
            label=f'Worst (Trial {worst_p[3]}): {worst_p[2]:+.2f}% SI')
    ang_b, tor_b = compute_torque_curve(best_p[0], best_p[1])
    ax.plot(ang_b, tor_b, color='green', linewidth=4,
            label=f'⭐ Best (Trial {best_p[3]}): {best_p[2]:+.2f}% SI')
    ax.axvspan(2, 20, alpha=0.1, color='red', label='PF Penalty Zone')
    ax.axvspan(-30, 2, alpha=0.05, color='green', label='DF Reward Zone')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(-10, color='green', linestyle=':', linewidth=2, alpha=0.7,
               label='DF Check (-10°)')
    ax.set_xlabel('Ankle Angle (deg)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Exo Torque (Nm)', fontsize=12, fontweight='bold')
    ax.set_title('Best vs Worst Trial | Torque-Angle Comparison',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10); ax.set_xlim(-30, 30)
    plt.tight_layout()
    p = output_dir / 'best_vs_worst_torque.png'
    plt.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ {p.name}")


def visualize_optimization(base_dir: str, subject: str, session: str) -> None:
    print("\n" + "=" * 70)
    print("PART 3: Bayesian Optimization progress")
    print("=" * 70)

    models_dir = Path(f'{base_dir}/sub-{subject}/ses-{session}/'
                      f'derivatives/hil_optimization/models')
    output_dir = Path(f'{base_dir}/sub-{subject}/ses-{session}/'
                      f'derivatives/hil_optimization/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    hil_results_path = Path(
        f'{base_dir}/sub-{subject}/ses-{session}/eeg/'
        f'sub-{subject}_ses-{session}_hil_results.csv')
    if not hil_results_path.exists():
        print(f"\n⚠️  HIL results not found: {hil_results_path}")
        return

    hil_results = pd.read_csv(hil_results_path)
    print(f"✅ Loaded {len(hil_results)} trials from hil_results.csv")

    # Infer param ranges from the data (handles any config, any subject)
    if len(hil_results) > 0:
        range_arr = np.array([
            [hil_results['R'].min(), hil_results['L0'].min()],
            [hil_results['R'].max(), hil_results['L0'].max()],
        ])
        # Pad a bit so rescaling doesn't compress everything to the edges
        for j in range(2):
            span = range_arr[1, j] - range_arr[0, j]
            range_arr[0, j] -= span * 0.05
            range_arr[1, j] += span * 0.05
    else:
        range_arr = np.array([[0.24, 0.30], [0.35, 0.40]])

    iter_folders = sorted([f for f in models_dir.glob('iter_*') if f.is_dir()],
                          key=lambda x: int(x.name.split('_')[1]))
    if not iter_folders:
        print(f"\n⚠️  No iteration folders in {models_dir}\n")
        return

    print(f"Found {len(iter_folders)} iterations")
    all_data = []
    for f in iter_folders:
        iter_num = int(f.name.split('_')[1])
        R, L0, asymm = plot_bo_iteration(f, iter_num, output_dir, hil_results, range_arr)
        if R is not None:
            all_data.append((iter_num, R, L0, asymm))

    if all_data:
        create_optimization_summary(all_data, output_dir)
        plot_best_vs_worst_torque(all_data, output_dir)

    print(f"\n✅ Plots in: {output_dir}\n")


# ===========================================================================
# PART 4 — Per-trial torque curves
# ===========================================================================

def plot_all_torque_curves(hil_results_path: Path, output_dir: Path) -> None:
    print("\n" + "=" * 70)
    print("PART 4: Per-trial torque curves")
    print("=" * 70)
    if not hil_results_path.exists():
        print(f"⚠️  {hil_results_path} not found")
        return

    hil_results = pd.read_csv(hil_results_path)
    n_trials = len(hil_results)
    if n_trials == 0:
        print("⚠️  Empty hil_results")
        return

    n_cols = 3
    n_rows = (n_trials + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4.3))
    axes = np.array(axes).reshape(-1)

    best_idx = hil_results['cost'].apply(abs).idxmin()
    best_trial = hil_results.loc[best_idx, 'trial']

    # Detect exploration count
    exp_trials = sum(1 for p in hil_results['phase'] if 'xploration' in str(p))
    if exp_trials == 0:
        exp_trials = 5   # default

    for idx in range(n_trials):
        ax = axes[idx]
        row = hil_results.iloc[idx]
        trial_num = int(row['trial'])
        R, L0, asymm = float(row['R']), float(row['L0']), float(row['cost'])
        angles, torques = compute_torque_curve(R, L0, angle_min=-30.0, angle_max=30.0)

        color = '#e74c3c' if trial_num <= exp_trials else '#3498db'
        phase = 'Exploration' if trial_num <= exp_trials else 'BO'
        linewidth = 2
        is_best = trial_num == best_trial
        if is_best:
            color = 'gold'; linewidth = 3
            for spine in ax.spines.values():
                spine.set_edgecolor('gold'); spine.set_linewidth(3)

        ax.plot(angles, torques, color=color, linewidth=linewidth)
        ax.axvspan(2, 20, alpha=0.15, color='red', zorder=0)
        ax.axvspan(-30, 2, alpha=0.08, color='green', zorder=0)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(-10, color='green', linestyle=':', linewidth=1.5, alpha=0.6)

        prefix = '⭐ BEST ⭐\n' if is_best else ''
        ax.set_title(
            f'{prefix}Trial {trial_num} ({phase})\n'
            f'R={R:.3f}m, L₀={L0:.3f}m\nSI: {asymm:+.2f}%',
            fontsize=9, fontweight='bold' if is_best else 'normal',
            color='darkgoldenrod' if is_best else 'black', pad=6)
        ax.set_xlim(-30, 30)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        if idx // n_cols == n_rows - 1: ax.set_xlabel('Angle (deg)', fontsize=9)
        if idx % n_cols == 0: ax.set_ylabel('Torque (Nm)', fontsize=9)
        ax.tick_params(labelsize=8)

    # Hide any unused axes
    for i in range(n_trials, len(axes)):
        axes[i].axis('off')

    fig.suptitle(
        f'All {n_trials} Trial Torque Curves | '
        f'Exploration: Red | BO: Blue | Best: Gold',
        fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])

    for ext in ['png', 'pdf']:
        p = output_dir / f'all_trials_torque_curves.{ext}'
        plt.savefig(p, dpi=300 if ext == 'png' else None,
                    bbox_inches='tight', facecolor='white')
        print(f"  ✅ {p.name}")
    plt.close()


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> int:
    args = parse_args()

    data_dir = f'{args.base_dir}/sub-{args.subject}/ses-{args.session}/eeg'
    opt_dir = (f'{args.base_dir}/sub-{args.subject}/ses-{args.session}/'
               f'derivatives/hil_optimization/visualizations')

    print("\n" + "🎯" * 35)
    print("HITLO_Symmetry — Post-hoc Experiment Analysis")
    print("🎯" * 35)
    print(f"\nSubject: {args.subject}   Session: {args.session}")
    print(f"Base dir: {args.base_dir}")
    print(f"Trim: {args.trim}s from each end of each trial\n")

    # Part 1: XDF → CSV
    if not args.no_xdf_processing:
        df = process_xdf_files(data_dir, trim_s=args.trim)
    else:
        csv = f'{data_dir}/master_gait_data.csv'
        if Path(csv).exists():
            df = pd.read_csv(csv)
            print(f"Reusing existing {csv} ({len(df)} rows)")
        else:
            print(f"⚠️  --no-xdf-processing set but {csv} doesn't exist")
            df = pd.DataFrame()

    # Part 2: Timeline plot
    if not args.no_timeline_plot and len(df) > 0:
        create_timeline_plot(df, data_dir)

    # Part 3: BO iteration plots
    if not args.no_optimization_plots:
        visualize_optimization(args.base_dir, args.subject, args.session)

    # Part 4: Torque curve grid
    if not args.no_torque_plots:
        hil_results_path = Path(
            f'{data_dir}/sub-{args.subject}_ses-{args.session}_hil_results.csv')
        Path(opt_dir).mkdir(parents=True, exist_ok=True)
        plot_all_torque_curves(hil_results_path, Path(opt_dir))

    print("=" * 70)
    print("✅ Analysis complete")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
