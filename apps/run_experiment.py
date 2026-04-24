"""
apps/run_experiment.py — Streamlit UI for HITLO_Symmetry live BO trials.

This is the clinician-facing tool. It:
  - Shows live Polar H10 streams so you can confirm sensors before each trial
  - Displays the (R, L₀) parameters the BO wants you to set this trial
  - Shows the predicted torque curve for those parameters
  - After each trial, analyzes the XDF and shows a QC plot of heel strikes
  - Tracks progress, cost, and the GP cost surface across trials
  - Auto-checkpoints so a crash doesn't lose your session

All detection logic lives in hitlo/ — this script is just the orchestration
and visualization layer.

Run with:
    streamlit run apps/run_experiment.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

# Make hitlo importable when running streamlit from the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hitlo.hil_exo import HIL_Exo
from hitlo.cost import (
    SymmetryCost, compute_torque_curve, compute_spring_penalty,
)
from hitlo.detection import detect_heelstrikes_full, DetectionConfig
from hitlo.io import load_both_polar_streams, trial_filename
from hitlo.symmetry import (
    compute_step_times, compute_symmetry_index, trim_peaks,
)


# ===========================================================================
# Streamlit setup + session state
# ===========================================================================

st.set_page_config(
    page_title="HITLO_Symmetry Experiment",
    page_icon="🦾",
    layout="wide",
)

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.current_trial = 0
    st.session_state.hil = None
    st.session_state.cost_extractor = None
    st.session_state.results = []
    st.session_state.config = None
    st.session_state.lsl_inlet_left = None
    st.session_state.lsl_inlet_right = None
    st.session_state.live_data_left = {'time': [], 'x': [], 'y': [], 'z': []}
    st.session_state.live_data_right = {'time': [], 'x': [], 'y': [], 'z': []}
    st.session_state.max_live_points = 1000


# ===========================================================================
# Config + checkpoint persistence
# ===========================================================================

def load_config() -> dict:
    for candidate in ['config/exo_symmetry_config.yml',
                      'exo_symmetry_config.yml']:
        if os.path.exists(candidate):
            with open(candidate) as f:
                return yaml.safe_load(f)
    return None


def _checkpoint_path(config) -> str:
    subject = config['Subject']['id']
    session = config['Subject']['session']
    base_dir = config['Subject']['base_dir']
    deriv = os.path.join(base_dir, f"sub-{subject}", f"ses-{session}",
                         "derivatives", "hil_optimization")
    os.makedirs(deriv, exist_ok=True)
    return os.path.join(deriv, f"sub-{subject}_ses-{session}_checkpoint.json")


def save_checkpoint() -> None:
    try:
        hil = st.session_state.hil
        config = st.session_state.config
        ckpt = {
            'current_trial': st.session_state.current_trial,
            'results': st.session_state.results,
            'x': hil.x.tolist(),
            'x_opt': hil.x_opt.tolist() if len(hil.x_opt) > 0 else [],
            'y_opt': hil.y_opt.tolist() if len(hil.y_opt) > 0 else [],
            'n': hil.n,
        }
        path = _checkpoint_path(config)
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(ckpt, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


def load_checkpoint(config) -> dict:
    try:
        path = _checkpoint_path(config)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return None


def delete_checkpoint(config) -> None:
    try:
        path = _checkpoint_path(config)
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ===========================================================================
# LSL live streaming
# ===========================================================================

def connect_to_lsl() -> bool:
    if (st.session_state.lsl_inlet_left is not None and
            st.session_state.lsl_inlet_right is not None):
        return True
    try:
        from pylsl import StreamInlet, resolve_streams
        streams = resolve_streams()
        for s in streams:
            if s.name() == 'polar accel left' and st.session_state.lsl_inlet_left is None:
                st.session_state.lsl_inlet_left = StreamInlet(s)
            if s.name() == 'polar accel right' and st.session_state.lsl_inlet_right is None:
                st.session_state.lsl_inlet_right = StreamInlet(s)
    except Exception:
        pass
    return (st.session_state.lsl_inlet_left is not None and
            st.session_state.lsl_inlet_right is not None)


def update_live_data(inlet, store) -> None:
    if inlet is None:
        return
    try:
        samples, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=100)
        if samples:
            for i, sample in enumerate(samples):
                store['time'].append(timestamps[i])
                store['x'].append(sample[0])
                store['y'].append(sample[1])
                store['z'].append(sample[2])
            max_pts = st.session_state.max_live_points
            for key in store:
                if len(store[key]) > max_pts:
                    store[key] = store[key][-max_pts:]
    except Exception:
        pass


# ===========================================================================
# System initialization
# ===========================================================================

def initialize_system(fresh_start: bool = False) -> Tuple[bool, bool]:
    st.session_state.config = load_config()
    if st.session_state.config is None:
        st.error("Config file not found (expected config/exo_symmetry_config.yml)!")
        return False, False

    config = st.session_state.config
    subject = config['Subject']['id']
    session = config['Subject']['session']
    base_dir = config['Subject']['base_dir']
    eeg_dir = os.path.join(base_dir, f"sub-{subject}", f"ses-{session}", "eeg")

    deriv_base = os.path.join(base_dir, f"sub-{subject}", f"ses-{session}",
                              "derivatives", "hil_optimization")
    config['Optimization']['model_save_path'] = os.path.join(deriv_base, "models") + "/"
    config['Optimization']['result_save_path'] = os.path.join(deriv_base, "results") + "/"
    os.makedirs(config['Optimization']['model_save_path'], exist_ok=True)
    os.makedirs(config['Optimization']['result_save_path'], exist_ok=True)
    os.makedirs(eeg_dir, exist_ok=True)

    opt = config['Optimization']
    signed = config['Cost'].get('signed', False)
    trim_s = config['Cost'].get('trim_seconds', 3.0)

    st.session_state.cost_extractor = SymmetryCost(
        trial_data_dir=eeg_dir,
        subject_id=subject,
        session=session,
        lambda_pf=opt.get('lambda_pf', 0.01),
        mu_df=opt.get('mu_df', 0.005),
        pf_zone=tuple(opt.get('pf_zone_deg', [2.0, 20.0])),
        df_angle=opt.get('df_check_angle_deg', -10.0),
        signed=signed,
        trim_seconds=trim_s,
    )

    st.session_state.hil = HIL_Exo(
        st.session_state.config, st.session_state.cost_extractor)

    ckpt = None if fresh_start else load_checkpoint(config)
    resumed = False
    if ckpt is not None:
        try:
            hil = st.session_state.hil
            hil.x = np.array(ckpt['x'])
            hil.x_opt = np.array(ckpt['x_opt']) if ckpt['x_opt'] else np.array([])
            hil.y_opt = np.array(ckpt['y_opt']) if ckpt['y_opt'] else np.array([])
            hil.n = ckpt['n']
            st.session_state.results = ckpt['results']
            st.session_state.current_trial = ckpt['current_trial']
            resumed = True
        except Exception as e:
            st.warning(f"⚠️ Checkpoint found but could not load ({e}). Starting fresh.")
            ckpt = None

    if ckpt is None:
        st.session_state.hil._generate_initial_parameters()
        st.session_state.results = []
        st.session_state.current_trial = 0
        delete_checkpoint(config)

    st.session_state.initialized = True
    return True, resumed


# ===========================================================================
# Trial analysis (uses hitlo.cost.SymmetryCost)
# ===========================================================================

def check_file_exists(trial_num: int) -> bool:
    trial_dir = st.session_state.cost_extractor.trial_data_dir
    fname = trial_filename(
        st.session_state.config['Subject']['id'],
        st.session_state.config['Subject']['session'],
        trial_num,
    )
    return os.path.exists(os.path.join(trial_dir, fname))


def analyze_current_trial() -> bool:
    trial_num = st.session_state.current_trial + 1
    config = st.session_state.config
    fname = trial_filename(config['Subject']['id'], config['Subject']['session'],
                           trial_num)

    if not check_file_exists(trial_num):
        st.error(f"❌ File not found: {fname}")
        return False

    hil = st.session_state.hil
    params = hil.x[hil.n]

    st.session_state.cost_extractor.set_params(R=params[0], L0=params[1])
    cost = st.session_state.cost_extractor.extract_cost_from_file(
        trial_num=trial_num, filename=fname)

    if cost is None or np.isnan(cost):
        st.error("❌ Cost extraction failed!")
        return False

    if len(hil.x_opt) < 1:
        hil.x_opt = np.array([params])
        hil.y_opt = np.array([cost])
    else:
        hil.x_opt = np.concatenate((hil.x_opt, np.array([params])))
        hil.y_opt = np.concatenate((hil.y_opt, np.array([cost])))

    n_exploration = config['Optimization']['n_exploration']
    signed = config['Cost'].get('signed', False)
    phase = ("Exploration (LHS)" if trial_num <= n_exploration
             else "Bayesian Optimization")

    st.session_state.results.append({
        'trial': trial_num,
        'R': params[0], 'L0': params[1],
        'cost': cost, 'phase': phase, 'signed': signed,
        'is_best': np.argmin(np.abs(hil.y_opt)) == len(hil.y_opt) - 1,
    })

    # Atomic CSV save
    try:
        subject = config['Subject']['id']
        session = config['Subject']['session']
        base_dir = config['Subject']['base_dir']
        save_dir = os.path.join(base_dir, f"sub-{subject}", f"ses-{session}", "eeg")
        save_path = os.path.join(
            save_dir, f"sub-{subject}_ses-{session}_hil_results.csv")
        tmp_path = save_path + '.tmp'
        pd.DataFrame(st.session_state.results).to_csv(tmp_path, index=False)
        os.replace(tmp_path, save_path)
    except Exception:
        pass

    save_checkpoint()
    hil.n += 1

    n_steps = config['Optimization']['n_steps']
    if hil.n >= n_exploration and hil.n < n_steps:
        if config['Optimization']['normalize']:
            norm_x = hil._normalize_x(hil.x_opt)
            norm_y = hil._mean_normalize_y(hil.y_opt)
            raw = hil.BO.run(
                norm_x.reshape(len(hil.x_opt), -1),
                norm_y.reshape(len(hil.x_opt), 1))
            raw = hil._denormalize_x(raw)
        else:
            raw = hil.BO.run(
                hil.x_opt.reshape(len(hil.x_opt), -1),
                (-np.abs(hil.y_opt) if signed else -hil.y_opt).reshape(len(hil.y_opt), 1))
        new_parameter = hil._get_safe_bo_suggestion(raw)
        hil.x = np.concatenate((
            hil.x, new_parameter.reshape(1, config['Optimization']['n_parms'])
        ), axis=0)

    st.session_state.current_trial += 1
    return True


# ===========================================================================
# Plots
# ===========================================================================

def plot_torque_curve(R: float, L0: float) -> go.Figure:
    angles, torques = compute_torque_curve(R, L0, angle_min=-30.0, angle_max=30.0)
    opt = st.session_state.config['Optimization']
    pf_zone = opt.get('pf_zone_deg', [2.0, 20.0])
    df_check_angle = opt.get('df_check_angle_deg', -10.0)

    fig = go.Figure()
    fig.add_vrect(x0=pf_zone[0], x1=pf_zone[1], fillcolor="red", opacity=0.1,
                  layer="below", line_width=0,
                  annotation_text=f"⚠️ PF Zone {pf_zone[0]}°–{pf_zone[1]}°",
                  annotation_position="top left", annotation_font_size=11)
    fig.add_vrect(x0=-30, x1=pf_zone[0], fillcolor="green", opacity=0.07,
                  layer="below", line_width=0,
                  annotation_text=f"✅ DF Zone (want peak @ {df_check_angle}°)",
                  annotation_position="top right", annotation_font_size=11)
    fig.add_trace(go.Scatter(x=angles, y=torques, mode='lines', name='Exo Torque',
                             line=dict(color='royalblue', width=3)))
    df_torque_at_peak = float(np.interp(df_check_angle, angles, torques))
    fig.add_trace(go.Scatter(x=[df_check_angle], y=[df_torque_at_peak], mode='markers',
                             name=f'Peak DF @ {df_check_angle}°: {df_torque_at_peak:.1f} Nm',
                             marker=dict(color='green', size=12, symbol='star')))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color="black", line_width=1,
                  annotation_text="Neutral", annotation_position="bottom right")
    fig.update_layout(
        title=f"Torque-Angle Curve  |  R = {R:.4f} m,  L₀ = {L0:.4f} m",
        xaxis_title="Ankle Angle (deg)  [DF ← 0 → PF]",
        yaxis_title="Exo Torque (Nm)",
        height=380, margin=dict(l=50, r=50, t=60, b=50),
        showlegend=True, hovermode='x unified',
        xaxis=dict(range=[-30, 30]),
    )
    return fig


def plot_gp_surface():
    import torch
    hil = st.session_state.hil
    if hil.n < st.session_state.config['Optimization']['n_exploration'] + 1:
        return None
    if hil.BO.model is None:
        return None

    config = st.session_state.config
    range_ = np.array(list(config['Optimization']['range'])).reshape(2, 2)
    R_min, L0_min = range_[0]
    R_max, L0_max = range_[1]
    n_grid = 40
    R_grid = np.linspace(R_min, R_max, n_grid)
    L0_grid = np.linspace(L0_min, L0_max, n_grid)
    RR, LL = np.meshgrid(R_grid, L0_grid)
    grid_pts = np.column_stack([RR.ravel(), LL.ravel()])

    if config['Optimization']['normalize']:
        grid_norm = (grid_pts - range_[0]) / (range_[1] - range_[0])
    else:
        grid_norm = grid_pts

    hil.BO.model.eval()
    hil.BO.likelihood.eval()
    with torch.no_grad():
        pred = hil.BO.likelihood(hil.BO.model(
            torch.tensor(grid_norm, dtype=torch.float64)))
        mean = pred.mean.cpu().numpy()
        std = pred.variance.sqrt().cpu().numpy()

    if config['Optimization']['normalize']:
        y_obs_mean = np.mean(hil.y_opt)
        y_obs_std = np.std(hil.y_opt) if np.std(hil.y_opt) > 0 else 1.0
        mean_display = (-mean * y_obs_std) + y_obs_mean
    else:
        y_obs_std = 1.0
        mean_display = -mean

    mean_display = mean_display.reshape(n_grid, n_grid)
    std_display = (std * y_obs_std).reshape(n_grid, n_grid)
    x_obs = hil.x_opt
    y_obs = hil.y_opt
    best_idx = np.argmin(np.abs(y_obs))

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=R_grid, y=L0_grid, z=mean_display,
        colorscale='RdYlGn_r', opacity=0.85, name='GP Mean',
        colorbar=dict(title='Predicted Cost', x=1.02), showscale=True))
    fig.add_trace(go.Surface(
        x=R_grid, y=L0_grid, z=mean_display + std_display,
        colorscale='Blues', opacity=0.18, showscale=False, name='+1 Std'))
    fig.add_trace(go.Surface(
        x=R_grid, y=L0_grid, z=mean_display - std_display,
        colorscale='Blues', opacity=0.18, showscale=False, name='-1 Std'))

    if len(x_obs) > 0:
        mask = np.ones(len(x_obs), dtype=bool)
        mask[best_idx] = False
        if mask.any():
            fig.add_trace(go.Scatter3d(
                x=x_obs[mask, 0], y=x_obs[mask, 1], z=y_obs[mask],
                mode='markers', name='Observed Trials',
                marker=dict(size=6, color='royalblue',
                            line=dict(color='white', width=1))))
        fig.add_trace(go.Scatter3d(
            x=[x_obs[best_idx, 0]], y=[x_obs[best_idx, 1]],
            z=[y_obs[best_idx]], mode='markers',
            name=f'Best (Cost={y_obs[best_idx]:.3f})',
            marker=dict(size=12, color='gold', symbol='diamond',
                        line=dict(color='darkorange', width=2))))

    fig.update_layout(
        title=f'GP Cost Surface  |  R [{R_min}–{R_max}]  L₀ [{L0_min}–{L0_max}]',
        scene=dict(xaxis_title='R (m)', yaxis_title='L₀ (m)',
                   zaxis_title='Predicted Cost',
                   camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))),
        height=650, margin=dict(l=0, r=0, t=60, b=0), showlegend=True,
    )
    return fig


def plot_progress():
    if not st.session_state.results:
        return None
    df = pd.DataFrame(st.session_state.results)
    colors = ['gold' if b else 'lightblue' for b in df['is_best']]
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Cost vs Trial', 'Parameters vs Trial'),
                        vertical_spacing=0.15, row_heights=[0.6, 0.4])
    fig.add_trace(go.Scatter(
        x=df['trial'], y=df['cost'], mode='lines+markers', name='Cost',
        marker=dict(size=12, color=colors, line=dict(width=2, color='darkblue')),
        line=dict(color='royalblue', width=2)), row=1, col=1)
    fig.add_hline(y=df['cost'].min(), line_dash="dash", line_color="red",
                  annotation_text=f"Best: {df['cost'].min():.2f}", row=1, col=1)
    fig.add_trace(go.Scatter(x=df['trial'], y=df['R'], mode='lines+markers',
                             name='R', marker=dict(size=8),
                             line=dict(width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['trial'], y=df['L0'], mode='lines+markers',
                             name='L₀', marker=dict(size=8),
                             line=dict(width=2)), row=2, col=1)
    fig.update_xaxes(title_text="Trial", row=2, col=1)
    fig.update_yaxes(title_text="Cost", row=1, col=1)
    fig.update_yaxes(title_text="Parameter Value", row=2, col=1)
    fig.update_layout(height=700, showlegend=True, hovermode='x unified')
    return fig


# ===========================================================================
# Heel-strike QC (uses hitlo.detection — always matches the BO cost)
# ===========================================================================

def analyze_trial_for_qc(xdf_path: str, cfg: DetectionConfig,
                         trim_seconds: float) -> dict:
    """Full trial analysis for QC display. Uses the SAME detection pipeline
    as the BO cost function, so what you see is what got scored."""
    left, right = load_both_polar_streams(xdf_path)
    if left is None or right is None:
        return None

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

    # CVs
    l_cv = (np.std(np.diff(l_times)) / np.mean(np.diff(l_times))
            if len(l_times) > 1 else np.nan)
    r_cv = (np.std(np.diff(r_times)) / np.mean(np.diff(r_times))
            if len(r_times) > 1 else np.nan)

    # Symmetry
    r_steps, l_steps = compute_step_times(l_times, r_times)
    n = min(len(r_steps), len(l_steps))
    if n >= 2:
        r_s, l_s = r_steps[:n], l_steps[:n]
        si_signed, _ = compute_symmetry_index(r_s, l_s, signed=True)
        si_unsigned, _ = compute_symmetry_index(r_s, l_s, signed=False)
        stride_time = float(r_s.mean() + l_s.mean())
        r_step_mean = float(r_s.mean())
        l_step_mean = float(l_s.mean())
    else:
        si_signed = si_unsigned = stride_time = np.nan
        r_step_mean = l_step_mean = np.nan

    drift_pct = abs(left.actual_fs - right.actual_fs) / 200.0 * 100.0

    return dict(
        left=left, right=right,
        left_result=left_result, right_result=right_result,
        l_times=l_times, r_times=r_times,
        t_lo=t_lo, t_hi=t_hi,
        l_cv=l_cv, r_cv=r_cv, drift_pct=drift_pct,
        si_signed=si_signed, si_unsigned=si_unsigned,
        stride_time=stride_time,
        r_step_mean=r_step_mean, l_step_mean=l_step_mean,
        cfg=cfg,
    )


def plot_heelstrikes_last_trial(xdf_path: str, cfg: DetectionConfig,
                                trim_seconds: float):
    qc = analyze_trial_for_qc(xdf_path, cfg, trim_seconds)
    if qc is None:
        return None, ["Could not load XDF or streams missing."]

    warnings = []

    if qc['drift_pct'] > 1.0:
        warnings.append(
            f"⚠️ **Clock drift {qc['drift_pct']:.2f}%** between sensors "
            f"(L={qc['left'].actual_fs:.1f} Hz, R={qc['right'].actual_fs:.1f} Hz). "
            f"LSL timestamps handle this — just informational.")

    n_l = len(qc['l_times'])
    n_r = len(qc['r_times'])
    if n_l < 10 or n_r < 10:
        warnings.append(
            f"🚨 **Low heel strike count after trim** — L={n_l}, R={n_r}. "
            f"Trial may be too short or detection is failing.")
    if abs(n_l - n_r) > 3:
        warnings.append(
            f"🚨 **Heel strike count mismatch** — L={n_l} vs R={n_r}.")
    if not np.isnan(qc['l_cv']) and qc['l_cv'] > 0.25:
        warnings.append(f"🚨 **Left CV = {qc['l_cv']:.3f}** (> 0.25) — erratic timing.")
    if not np.isnan(qc['r_cv']) and qc['r_cv'] > 0.25:
        warnings.append(f"🚨 **Right CV = {qc['r_cv']:.3f}** (> 0.25) — erratic timing.")
    if not np.isnan(qc['si_signed']) and abs(qc['si_signed']) > 40:
        warnings.append(
            f"🚨 **Symmetry = {qc['si_signed']:+.1f}%** — unrealistically large. "
            f"Possible sensor label swap or severe detection error.")

    # 3-panel plot
    t0 = min(qc['left'].timestamps[0], qc['right'].timestamps[0])
    t_left = qc['left'].timestamps - t0
    t_right = qc['right'].timestamps - t0
    trim_lo_rel = qc['t_lo'] - t0
    trim_hi_rel = qc['t_hi'] - t0
    max_t = max(t_left[-1], t_right[-1])

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=(
            'LEFT raw magnitude (▼ = heel strike, × = rejected candidate)',
            'RIGHT raw magnitude (▼ = heel strike, × = rejected candidate)',
            'Jerk z-score overlay',
        ),
        vertical_spacing=0.08, row_heights=[0.33, 0.33, 0.34]
    )

    def shade_clusters(row_idx, clusters, ts):
        for (cstart, cend) in clusters:
            if cstart >= len(ts) or cend >= len(ts):
                continue
            x0, x1 = ts[cstart], ts[cend]
            if cstart == cend:
                fig.add_vrect(x0=x0 - 0.04, x1=x1 + 0.04,
                              fillcolor='limegreen', opacity=0.15,
                              layer='below', line_width=0, row=row_idx, col=1)
            else:
                fig.add_vrect(x0=x0, x1=x1, fillcolor='salmon', opacity=0.22,
                              layer='below', line_width=0, row=row_idx, col=1)

    # Row 1: LEFT raw magnitude
    shade_clusters(1, qc['left_result'].cluster_info, t_left)
    fig.add_trace(go.Scatter(x=t_left, y=qc['left_result'].magnitude,
                             mode='lines', name='L magnitude',
                             line=dict(color='steelblue', width=1.0),
                             opacity=0.75), row=1, col=1)
    baseline_l = float(np.median(qc['left_result'].magnitude))
    fig.add_hline(y=baseline_l, line_dash='dashdot', line_color='gray',
                  annotation_text=f'baseline ({baseline_l:.0f})', row=1, col=1)
    if len(qc['left_result'].heel_strike_indices) > 0:
        acc = qc['left_result'].heel_strike_indices
        safe = acc[acc < len(qc['left_result'].magnitude)]
        fig.add_trace(go.Scatter(
            x=t_left[safe], y=qc['left_result'].magnitude[safe],
            mode='markers', name=f"L accepted ({len(acc)})",
            marker=dict(symbol='triangle-down', size=10, color='navy')
        ), row=1, col=1)
    if len(qc['left_result'].rejected_peaks) > 0:
        rej = qc['left_result'].rejected_peaks
        safe = rej[rej < len(qc['left_result'].magnitude)]
        fig.add_trace(go.Scatter(
            x=t_left[safe], y=qc['left_result'].magnitude[safe],
            mode='markers', name=f"L rejected ({len(rej)})",
            marker=dict(symbol='x', size=9, color='gray', line=dict(width=1.5))
        ), row=1, col=1)

    # Row 2: RIGHT raw magnitude
    shade_clusters(2, qc['right_result'].cluster_info, t_right)
    fig.add_trace(go.Scatter(x=t_right, y=qc['right_result'].magnitude,
                             mode='lines', name='R magnitude',
                             line=dict(color='tomato', width=1.0),
                             opacity=0.75), row=2, col=1)
    baseline_r = float(np.median(qc['right_result'].magnitude))
    fig.add_hline(y=baseline_r, line_dash='dashdot', line_color='gray',
                  annotation_text=f'baseline ({baseline_r:.0f})', row=2, col=1)
    if len(qc['right_result'].heel_strike_indices) > 0:
        acc = qc['right_result'].heel_strike_indices
        safe = acc[acc < len(qc['right_result'].magnitude)]
        fig.add_trace(go.Scatter(
            x=t_right[safe], y=qc['right_result'].magnitude[safe],
            mode='markers', name=f"R accepted ({len(acc)})",
            marker=dict(symbol='triangle-down', size=10, color='darkred')
        ), row=2, col=1)
    if len(qc['right_result'].rejected_peaks) > 0:
        rej = qc['right_result'].rejected_peaks
        safe = rej[rej < len(qc['right_result'].magnitude)]
        fig.add_trace(go.Scatter(
            x=t_right[safe], y=qc['right_result'].magnitude[safe],
            mode='markers', name=f"R rejected ({len(rej)})",
            marker=dict(symbol='x', size=9, color='gray', line=dict(width=1.5))
        ), row=2, col=1)

    # Row 3: jerk z-score overlay
    fig.add_trace(go.Scatter(x=t_left, y=qc['left_result'].jerk_z,
                             mode='lines', name='L jerk z',
                             line=dict(color='steelblue', width=0.8),
                             opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=t_right, y=qc['right_result'].jerk_z,
                             mode='lines', name='R jerk z',
                             line=dict(color='tomato', width=0.8),
                             opacity=0.7), row=3, col=1)
    fig.add_hline(y=qc['cfg'].strict_thresh, line_dash='dash', line_color='green',
                  annotation_text=f"{qc['cfg'].strict_thresh} SD strict", row=3, col=1)
    fig.add_hline(y=qc['cfg'].recovery_thresh, line_dash='dot', line_color='orange',
                  annotation_text=f"{qc['cfg'].recovery_thresh} SD recovery",
                  row=3, col=1)

    # Trim shading on all rows
    if trim_seconds > 0:
        for row in (1, 2, 3):
            fig.add_vrect(x0=0, x1=trim_lo_rel, fillcolor='gray', opacity=0.18,
                          layer='below', line_width=0, row=row, col=1)
            fig.add_vrect(x0=trim_hi_rel, x1=max_t, fillcolor='gray', opacity=0.18,
                          layer='below', line_width=0, row=row, col=1)

    # Title
    if not np.isnan(qc['si_signed']):
        subtitle = (f"SI = {qc['si_signed']:+.2f}% signed  |  "
                    f"{qc['si_unsigned']:.2f}% unsigned  |  "
                    f"stride = {qc['stride_time']:.3f}s  |  "
                    f"L step = {qc['l_step_mean']:.3f}s, "
                    f"R step = {qc['r_step_mean']:.3f}s")
    else:
        subtitle = "Not enough step pairs to compute symmetry"

    title = (f"Heel Strike QC — "
             f"L: {len(qc['left_result'].all_candidates)} candidates "
             f"→ {len(qc['left_result'].heel_strike_indices)} heel strikes  |  "
             f"R: {len(qc['right_result'].all_candidates)} candidates "
             f"→ {len(qc['right_result'].heel_strike_indices)} heel strikes<br>"
             f"<sup>{subtitle}  |  pink = multi-peak cluster, "
             f"green = singleton cluster</sup>")

    fig.update_layout(
        title=title, height=720, margin=dict(l=50, r=20, t=100, b=40),
        hovermode='x unified', showlegend=True)
    fig.update_yaxes(title_text='|a|', row=1, col=1)
    fig.update_yaxes(title_text='|a|', row=2, col=1)
    fig.update_yaxes(title_text='jerk z', row=3, col=1)
    fig.update_xaxes(title_text='LSL time (s, rel. to earliest start)',
                     row=3, col=1)

    return fig, warnings


# ===========================================================================
# Live sensor plot
# ===========================================================================

def plot_live_sensor(store, name: str, sample_rate: int):
    n = len(store['time'])
    if n < 10:
        return None
    times = np.array(store['time'])
    times = times - times[0]
    last_n = min(n, sample_rate * 5)
    times = times[-last_n:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=np.array(store['x'][-last_n:]),
                             name='X (lateral)', mode='lines',
                             line=dict(color='red', width=1.5)))
    fig.add_trace(go.Scatter(x=times, y=np.array(store['y'][-last_n:]),
                             name='Y (fwd-back)', mode='lines',
                             line=dict(color='green', width=1)))
    fig.add_trace(go.Scatter(x=times, y=np.array(store['z'][-last_n:]),
                             name='Z (vertical)', mode='lines',
                             line=dict(color='blue', width=1)))
    fig.update_layout(title=f"{name} — last 5s",
                      xaxis_title="Time (s)", yaxis_title="Accel (mg)",
                      height=220, margin=dict(l=40, r=20, t=40, b=40),
                      showlegend=True, uirevision=name)
    return fig


# ===========================================================================
# MAIN UI
# ===========================================================================

st.title("🦾 HITLO_Symmetry — Exoskeleton Optimization")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    _cfg_peek = load_config()
    _has_ckpt = (bool(_cfg_peek and os.path.exists(_checkpoint_path(_cfg_peek)))
                 if _cfg_peek else False)

    if _has_ckpt and not st.session_state.initialized:
        st.warning("⚠️ Checkpoint found!")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶️ Resume", use_container_width=True, type="primary"):
                ok, _ = initialize_system(fresh_start=False)
                if ok:
                    st.success("✅ Resumed!")
                    st.rerun()
        with col_b:
            if st.button("🆕 Fresh Start", use_container_width=True):
                ok, _ = initialize_system(fresh_start=True)
                if ok:
                    st.success("✅ Started fresh!")
                    st.rerun()
    else:
        if st.button("🔄 Initialize/Reset System", use_container_width=True):
            ok, _ = initialize_system(fresh_start=True)
            if ok:
                st.success("✅ System initialized!")
                st.rerun()
            else:
                st.error("❌ Initialization failed!")

    if st.session_state.initialized:
        config = st.session_state.config
        opt = config['Optimization']
        range_ = np.array(list(opt['range'])).reshape(2, 2)
        signed = config['Cost'].get('signed', False)
        trim_s = config['Cost'].get('trim_seconds', 3.0)
        st.info(f"""
        **Experiment Settings:**
        - Total Trials: {opt['n_steps']}
        - Exploration: {opt['n_exploration']} (LHS)
        - Symmetry: {'Signed (two sensors)' if signed else 'Unsigned (sternum)'}
        - Trim: {trim_s:.1f}s each end
        - R range: {range_[0,0]} → {range_[1,0]}
        - L₀ range: {range_[0,1]} → {range_[1,1]}
        - Trial Duration: {config['Cost']['time']}s
        """)
        st.markdown("---")
        if st.session_state.results:
            if st.button("💾 Export Results", use_container_width=True):
                df = pd.DataFrame(st.session_state.results)
                csv = df.to_csv(index=False)
                st.download_button(label="Download CSV", data=csv,
                                   file_name="hil_results.csv", mime="text/csv")

if not st.session_state.initialized:
    st.info("👈 Click **Initialize/Reset System** in the sidebar to begin")
    st.stop()

# Live sensor monitoring
st.subheader("📡 Live Polar H10 Sensors (Left + Right Shank)")

connect_to_lsl()


@st.fragment(run_every=5.0)
def live_sensor_fragment():
    for side in ['left', 'right']:
        inlet = st.session_state[f'lsl_inlet_{side}']
        if inlet is not None:
            try:
                inlet.pull_chunk(timeout=0.0, max_samples=1)
            except Exception:
                st.session_state[f'lsl_inlet_{side}'] = None

    connect_to_lsl()
    sample_rate = st.session_state.config['Cost']['sample_rate']

    col_l, col_r = st.columns(2)

    for side, col in [('left', col_l), ('right', col_r)]:
        inlet = st.session_state[f'lsl_inlet_{side}']
        store = st.session_state[f'live_data_{side}']
        with col:
            if inlet is not None:
                update_live_data(inlet, store)
                n = len(store['time'])
                st.success(f"✅ {side.capitalize()} connected  |  {n} samples")
                fig = plot_live_sensor(store, f'polar accel {side}', sample_rate)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"⏳ Collecting {side} data...")
            else:
                st.warning(f"⚠️ {side.capitalize()} sensor not found")
                if st.button(f"🔄 Reconnect {side}", key=f"reconnect_{side}"):
                    st.session_state[f'lsl_inlet_{side}'] = None
                    st.session_state[f'live_data_{side}'] = {
                        'time': [], 'x': [], 'y': [], 'z': []}
                    connect_to_lsl()
                    st.rerun()


live_sensor_fragment()

if st.session_state.initialized:
    missing = []
    if st.session_state.lsl_inlet_left is None:
        missing.append('LEFT')
    if st.session_state.lsl_inlet_right is None:
        missing.append('RIGHT')
    if missing:
        st.error(f"🚨 **SENSOR DISCONNECTED** — {', '.join(missing)} sensor(s) "
                 f"lost. Do NOT start a trial.")

st.markdown("---")

# Experiment section
hil = st.session_state.hil
trial_num = st.session_state.current_trial + 1
n_steps = st.session_state.config['Optimization']['n_steps']

if trial_num > n_steps:
    delete_checkpoint(st.session_state.config)
    st.success("🎉 **OPTIMIZATION COMPLETE!** 🎉")
    best_idx = np.argmin(np.abs(hil.y_opt))
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Best Cost", f"{hil.y_opt[best_idx]:.4f}")
    with c2:
        st.metric("Best R", f"{hil.x_opt[best_idx][0]:.4f}")
    with c3:
        st.metric("Best L₀", f"{hil.x_opt[best_idx][1]:.4f}")
    st.markdown("---")
    if (fig := plot_progress()):
        st.plotly_chart(fig, use_container_width=True)
    if (gp_fig := plot_gp_surface()):
        st.plotly_chart(gp_fig, use_container_width=True)
    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df, use_container_width=True, height=400)
    st.stop()

c1, c2 = st.columns([2, 1])
with c1:
    st.header(f"Trial {trial_num}/{n_steps}")
    n_exploration = st.session_state.config['Optimization']['n_exploration']
    if trial_num <= n_exploration:
        st.info(f"**Phase:** 🎲 Exploration (LHS) — "
                f"trial {trial_num} of {n_exploration}")
    else:
        st.success(f"**Phase:** 🧠 Bayesian Optimization — "
                   f"trial {trial_num - n_exploration} of {n_steps - n_exploration}")
with c2:
    progress = (trial_num - 1) / n_steps
    st.metric("Progress", f"{int(progress * 100)}%")
    st.progress(progress)

st.markdown("---")

signed = st.session_state.config["Cost"].get("signed", False)
if hil.n >= len(hil.x):
    if len(hil.x_opt) >= n_exploration:
        try:
            if st.session_state.config['Optimization']['normalize']:
                norm_x = hil._normalize_x(hil.x_opt)
                norm_y = hil._mean_normalize_y(hil.y_opt)
                n_obs = len(hil.x_opt)
                raw = hil.BO.run(norm_x.reshape(n_obs, -1),
                                 norm_y.reshape(n_obs, 1))
                raw = hil._denormalize_x(raw)
            else:
                n_obs = len(hil.x_opt)
                raw = hil.BO.run(
                    hil.x_opt.reshape(n_obs, -1),
                    (-np.abs(hil.y_opt) if signed else -hil.y_opt).reshape(n_obs, -1))
            new_parameter = hil._get_safe_bo_suggestion(raw)
            hil.x = np.concatenate((
                hil.x, new_parameter.reshape(
                    1, st.session_state.config['Optimization']['n_parms'])
            ), axis=0)
        except Exception as e:
            st.error(f"BO suggestion failed: {e}")
            st.stop()
    else:
        st.error("Parameter index out of bounds — please reinitialize.")
        st.stop()

params = hil.x[hil.n]

st.subheader("📋 Parameters to Enter into Computer 2")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"### R = `{params[0]:.4f}` m")
with c2:
    st.markdown(f"### L₀ = `{params[1]:.4f}` m")

st.subheader("📈 Predicted Torque-Angle Curve")
torque_fig = plot_torque_curve(R=params[0], L0=params[1])
st.plotly_chart(torque_fig, use_container_width=True)

angles_check, torques_check = compute_torque_curve(params[0], params[1])
pf_zone = st.session_state.config['Optimization'].get('pf_zone_deg', [2.0, 20.0])
pf_mask = (angles_check >= pf_zone[0]) & (angles_check <= pf_zone[1])
pf_rms = np.sqrt(np.mean(torques_check[pf_mask] ** 2))
df_check_angle = st.session_state.config['Optimization'].get(
    'df_check_angle_deg', -10.0)
df_torque_peak = float(np.interp(df_check_angle, angles_check, torques_check))
pen = compute_spring_penalty(
    params[0], params[1],
    lambda_pf=st.session_state.cost_extractor.lambda_pf,
    mu_df=st.session_state.cost_extractor.mu_df)
mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.metric("RMS Torque in PF Zone (want → 0)", f"{pf_rms:.2f} Nm")
with mc2:
    st.metric("Torque at Peak DF (want → max)", f"{df_torque_peak:.2f} Nm")
with mc3:
    st.metric("Shape Penalty", f"{pen:.4f}")

st.markdown("---")

config = st.session_state.config
st.subheader("📝 Instructions")
with st.expander("**Step-by-step guide**", expanded=True):
    st.markdown(f"""
    1. **Enter parameters** into Computer 2:
       - R = {params[0]:.4f}
       - L₀ = {params[1]:.4f}

    2. **Start LabRecorder:**
       - Click "Update" — check **polar accel left** and **polar accel right**
       - Set filename: `sub-{config['Subject']['id']}_ses-{config['Subject']['session']}_task-Default_run-{trial_num:03d}_eeg.xdf`
       - Set save location: `{st.session_state.cost_extractor.trial_data_dir}`
       - Click "Start"

    3. **Subject walks** for {config['Cost']['time']} seconds

    4. Click **Stop** in LabRecorder

    5. Click **Analyze Trial** below
    """)

st.markdown("---")

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    file_exists = check_file_exists(trial_num)
    if file_exists:
        st.success(f"✅ File found: run-{trial_num:03d}.xdf")
    else:
        st.warning(f"⏳ Waiting for: run-{trial_num:03d}.xdf")
with c2:
    if st.button("🔍 Check File", use_container_width=True):
        st.rerun()
with c3:
    if st.button("▶️ Analyze Trial", type="primary",
                 use_container_width=True, disabled=not file_exists):
        with st.spinner("Analyzing trial..."):
            if analyze_current_trial():
                st.success("✅ Trial analyzed!")
                st.rerun()
            else:
                st.error("❌ Analysis failed!")

st.markdown("---")

if st.session_state.results:
    st.subheader("📊 Current Results")
    signed = config['Cost'].get('signed', False)
    best_cost = min(st.session_state.results, key=lambda r: abs(r['cost']))['cost']
    latest_cost = st.session_state.results[-1]['cost']
    trials_done = len(st.session_state.results)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Latest Cost", f"{latest_cost:.4f}")
    with c2:
        st.metric("Best Cost", f"{best_cost:.4f}",
                  delta=f"{latest_cost - best_cost:.4f}", delta_color="inverse")
    with c3:
        st.metric("Trials Completed", f"{trials_done}/{n_steps}")

    if signed:
        st.caption("Signed symmetry: positive = left leg slower, "
                   "negative = right leg slower, zero = symmetric")

    # QC plot
    last_trial_num = st.session_state.results[-1]['trial']
    last_fn = trial_filename(config['Subject']['id'],
                             config['Subject']['session'], last_trial_num)
    last_fp = os.path.join(
        st.session_state.cost_extractor.trial_data_dir, last_fn)
    trim_s = config['Cost'].get('trim_seconds', 3.0)
    cfg = st.session_state.cost_extractor.detection_cfg
    hs_result = plot_heelstrikes_last_trial(last_fp, cfg, trim_seconds=trim_s)
    if hs_result is not None:
        hs_fig, hs_warnings = hs_result

        with st.expander(f"🦶 Heel Strike QC — Trial {last_trial_num}",
                         expanded=True):
            if hs_warnings:
                for w in hs_warnings:
                    if "🚨" in w:
                        st.error(w)
                    else:
                        st.warning(w)
            else:
                st.success("✅ All QC checks passed — detection looks clean.")

            if hs_fig is not None:
                st.plotly_chart(hs_fig, use_container_width=True)
                st.caption(
                    f"Detection: jerk z-score → cluster-keep-last → stance "
                    f"confirmation → {trim_s:.1f}s trim (gray shade). Each "
                    f"cluster emits one heel strike — the last peak above "
                    f"baseline whose post-peak window is near baseline "
                    f"(stance). Same pipeline as the BO cost.")

    if (fig := plot_progress()):
        st.plotly_chart(fig, use_container_width=True)

    if len(st.session_state.results) > n_exploration:
        st.subheader("🧠 GP Predicted Cost Surface")
        if (gp_fig := plot_gp_surface()):
            st.plotly_chart(gp_fig, use_container_width=True)

    st.subheader("📋 Trial History")
    df = pd.DataFrame(st.session_state.results)
    df_display = df.copy()
    df_display['R'] = df_display['R'].map('{:.4f}'.format)
    df_display['L0'] = df_display['L0'].map('{:.4f}'.format)
    df_display['cost'] = df_display['cost'].map('{:.4f}'.format)
    df_display['best'] = df_display['is_best'].map(lambda x: '⭐' if x else '')
    df_display = df_display[['trial', 'R', 'L0', 'cost', 'phase', 'best']]
    st.dataframe(df_display, use_container_width=True, height=300)

st.markdown("---")
st.caption("HITLO_Symmetry v2.0.0 | Built on HIL_toolkit (Kim lab)")
