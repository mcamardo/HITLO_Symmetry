"""
apps/hitlo_console.py — HITLO_Symmetry multi-page console.

UPDATED for (L0, attach) parameterization with R fixed at 0.28 m.

This is the clinician-facing tool. It:
  - Shows live Polar H10 streams so you can confirm sensors before each trial
  - Runs a BASELINE phase first (no-band "Pre" trials) to measure the
    subject's natural asymmetry, then computes the optimization target
    relative to that baseline
  - Displays the (L0, attach) parameters the BO wants you to set this trial
  - Shows the predicted torque curve for those parameters
  - After each trial, analyzes the XDF and shows a QC plot of heel strikes
  - Tracks progress, cost, and the GP cost surface across trials
  - Auto-checkpoints so a crash doesn't lose your session

PARADIGM (Cost.si_target in config, OR set live from baseline)
--------------------------------------------------------------
  - si_target =   0.0 → drive SI toward 0 (Aim 2 stroke, "minimize asymmetry")
  - si_target < 0      → drive SI toward a negative target (Aim 1 healthy,
                         "induce target asymmetry" via passive band)

CONFIG EDITOR (first page)
--------------------------
On launch, before initialization, the operator edits the experiment config
(subject, session, target paradigm, ranges, etc.) directly in the UI and
saves it back to config/exo_symmetry_config.yml. No hand-editing the YAML.

BASELINE-RELATIVE TARGETING (Aim 1)
-----------------------------------
For healthy subjects, the goal is to induce a fixed *displacement* from the
subject's own baseline, not a fixed absolute SI. The baseline phase:
  1. Pre run-001 = no-device familiarization trial — IGNORED (not analyzed).
  2. Pre run-002 = THE baseline trial (band slack / no perturbation).
     Its signed SI alone defines baseline_si (no averaging).
  3. Target is set to:  si_target = baseline_si - displacement
     (always pushed more negative, matching the left-side device geometry)
  4. The computed target is written into the live config and drives the BO.

Baseline files: task-Pre, run-001 (ignored), run-002 (the baseline)
Optimization files: task-Default, run-001, run-002, ...
(Different task tags -> they never collide.)

All detection logic lives in hitlo/ — this script is just the orchestration
and visualization layer.

Run with:
    streamlit run apps/hitlo_console.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

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


# Baseline trials use this BIDS task tag (LabRecorder Block/Task dropdown).
BASELINE_TASK = "Pre"
# Pre run-001 is a no-device familiarization trial that we IGNORE.
# Pre run-002 is THE baseline trial whose SI defines the target.
BASELINE_IGNORE_RUN = 1
BASELINE_TRIAL_RUN = 2

# Fixed anchor distance (no longer optimized)
R_FIXED = 0.28


# ===========================================================================
# Small helpers for the paradigm
# ===========================================================================

def _signed_mode(config: dict) -> bool:
    return config['Cost'].get('signed', False)


def _si_target(config: dict) -> float:
    """Read Cost.si_target from config, default 0.0 (legacy behavior)."""
    return float(config['Cost'].get('si_target', 0.0))


def _is_aim2(config: dict) -> bool:
    """Aim 2 stroke = drive to symmetry (target 0), no baseline phase."""
    return str(config['Cost'].get('aim', 'Aim 1')).startswith('Aim 2')


def _best_idx(y_opt: np.ndarray, config: dict) -> int:
    """Single source of truth for best-trial selection in the UI.

    Mirrors hitlo.hil_exo.HIL_Exo._best_so_far_idx so the UI ranking and
    the BO ranking agree.
    """
    if _signed_mode(config):
        return int(np.argmin(np.abs(y_opt - _si_target(config))))
    return int(np.argmin(y_opt))


def _distance_from_target(cost_value: float, config: dict) -> float:
    """|SI - si_target| in signed mode; |cost| otherwise."""
    if _signed_mode(config):
        return abs(cost_value - _si_target(config))
    return abs(cost_value)


# ===========================================================================
# Streamlit setup + session state
# ===========================================================================

# (page config set by navigation entry point below)

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
    # --- Baseline phase state ---
    st.session_state.baseline_done = False
    st.session_state.baseline_si = None     # SI from Pre run-002 (the baseline)
    st.session_state.baseline_displacement = None  # displacement used to lock
    # --- Config editor state ---
    st.session_state.config_saved = False   # has the operator saved config yet


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
            # Paradigm + baseline settings, so we can detect a config change
            # on resume and restore the baseline-derived target.
            'signed': _signed_mode(config),
            'si_target': _si_target(config),
            'baseline_done': st.session_state.baseline_done,
            'baseline_si': st.session_state.baseline_si,
            'baseline_displacement': st.session_state.baseline_displacement,
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
    signed = _signed_mode(config)
    si_target = _si_target(config)
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
        si_target=si_target,
        trim_seconds=trim_s,
    )

    st.session_state.hil = HIL_Exo(
        st.session_state.config, st.session_state.cost_extractor)

    ckpt = None if fresh_start else load_checkpoint(config)
    resumed = False
    if ckpt is not None:
        # Detect a paradigm change since last checkpoint — if the operator
        # flipped signed mode (Aim 1 vs Aim 2) without intending to resume,
        # refuse to resume rather than mix datasets. Note: si_target may have
        # been set from baseline, so we compare `signed` strictly but allow
        # si_target to be restored FROM the checkpoint below.
        ckpt_signed = ckpt.get('signed', None)
        if ckpt_signed is not None and ckpt_signed != signed:
            st.warning(
                f"⚠️ Checkpoint paradigm mismatch: checkpoint had "
                f"signed={ckpt_signed}; config has signed={signed}. "
                f"Refusing to resume — use 🆕 Fresh Start, or revert the "
                f"config to match the checkpoint."
            )
            return False, False
        try:
            hil = st.session_state.hil
            hil.x = np.array(ckpt['x'])
            hil.x_opt = np.array(ckpt['x_opt']) if ckpt['x_opt'] else np.array([])
            hil.y_opt = np.array(ckpt['y_opt']) if ckpt['y_opt'] else np.array([])
            hil.n = ckpt['n']
            st.session_state.results = ckpt['results']
            st.session_state.current_trial = ckpt['current_trial']

            # Restore baseline state + the baseline-derived target.
            st.session_state.baseline_done = ckpt.get('baseline_done', False)
            st.session_state.baseline_si = ckpt.get('baseline_si', None)
            st.session_state.baseline_displacement = ckpt.get(
                'baseline_displacement', None)
            ckpt_target = ckpt.get('si_target', None)
            if ckpt_target is not None:
                # The locked baseline target wins on resume.
                config['Cost']['si_target'] = float(ckpt_target)
                st.session_state.cost_extractor.si_target = float(ckpt_target)
                st.session_state.hil.si_target = float(ckpt_target)
                st.session_state.hil._bo_direction_str = (
                    f"|SI - {float(ckpt_target):+.1f}|")
            resumed = True
        except Exception as e:
            st.warning(f"⚠️ Checkpoint found but could not load ({e}). Starting fresh.")
            ckpt = None

    if ckpt is None:
        st.session_state.hil._generate_initial_parameters()
        st.session_state.results = []
        st.session_state.current_trial = 0
        # Reset baseline state on a fresh start.
        st.session_state.baseline_done = False
        st.session_state.baseline_si = None
        st.session_state.baseline_displacement = None
        delete_checkpoint(config)

    st.session_state.initialized = True
    return True, resumed


# ===========================================================================
# Baseline analysis (no-band "Pre" trials)
# ===========================================================================

def baseline_filename(run_num: int) -> str:
    """BIDS filename for a baseline (no-band) trial, task-Pre."""
    config = st.session_state.config
    return trial_filename(config['Subject']['id'],
                          config['Subject']['session'],
                          run_num, task=BASELINE_TASK)


def baseline_file_exists(run_num: int) -> bool:
    fp = os.path.join(st.session_state.cost_extractor.trial_data_dir,
                      baseline_filename(run_num))
    return os.path.exists(fp)


def analyze_baseline(run_num: int) -> Optional[float]:
    """Analyze a no-band baseline trial; return its signed SI (no penalty).

    Uses the SAME detection + symmetry pipeline as the optimization cost so
    the baseline SI is directly comparable to the trial SIs. Spring params are
    irrelevant for the baseline (band is slack), so we pass placeholder values
    — the penalty is disabled (lambda_pf/mu_df = 0) and unused here anyway.
    """
    fname = baseline_filename(run_num)
    fp = os.path.join(st.session_state.cost_extractor.trial_data_dir, fname)
    if not os.path.exists(fp):
        return None
    st.session_state.cost_extractor.set_params(L0=0.0, attach=0.0)
    analysis = st.session_state.cost_extractor.analyze_trial(
        trial_num=run_num, filename=fname, verbose=False)
    if analysis is None:
        return None
    return float(analysis.symmetry_index)


def compute_baseline_target(baseline_si: float, displacement: float) -> float:
    """Optimization target = baseline pushed `displacement`% more negative.

    baseline_si comes from the single Pre run-002 trial (run-001 is the
    ignored no-device familiarization). No averaging.

    Always-negative rule: matches the left-side LegExoNET device geometry,
    which perturbs gait toward SI < 0 (longer left step time). A subject who
    starts at +2% gets target -8%; a subject at -3% gets target -13%. Either
    way the *induced displacement* is a constant `displacement`% — that's the
    error-augmentation 'dose' held fixed across subjects.
    """
    return baseline_si - displacement


# ===========================================================================
# Config file editor (first page, before initialization)
# ===========================================================================

CONFIG_PATH_CANDIDATES = ['config/exo_symmetry_config.yml',
                          'exo_symmetry_config.yml']


def _config_path() -> str:
    for c in CONFIG_PATH_CANDIDATES:
        if os.path.exists(c):
            return c
    return CONFIG_PATH_CANDIDATES[0]  # default write location


def save_config_to_disk(cfg: dict) -> bool:
    """Write the edited config back to the YAML file."""
    try:
        path = _config_path()
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        os.replace(tmp, path)
        return True
    except Exception as e:
        st.error(f"Could not save config: {e}")
        return False


# ===========================================================================
# Trial analysis (optimization phase; uses hitlo.cost.SymmetryCost)
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

    # NEW: params are (L0, attach), not (R, L0)
    st.session_state.cost_extractor.set_params(L0=params[0], attach=params[1])
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
    signed = _signed_mode(config)
    si_target = _si_target(config)
    phase = ("Exploration (LHS)" if trial_num <= n_exploration
             else "Bayesian Optimization")

    # "is_best" is recomputed every trial against the configured target.
    best_idx = _best_idx(hil.y_opt, config)
    this_is_best = (best_idx == len(hil.y_opt) - 1)
    for i, r in enumerate(st.session_state.results):
        r['is_best'] = (i == best_idx)

    # NEW: Store L0 and attach instead of R and L0
    st.session_state.results.append({
        'trial': trial_num,
        'L0': params[0], 'attach': params[1],
        'cost': cost,
        'dist_from_target': _distance_from_target(cost, config),
        'phase': phase,
        'signed': signed,
        'si_target': si_target,
        'baseline_si': st.session_state.baseline_si,
        'is_best': this_is_best,
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
            if signed:
                y_for_bo = -np.abs(hil.y_opt - si_target)
            else:
                y_for_bo = -hil.y_opt
            raw = hil.BO.run(
                hil.x_opt.reshape(len(hil.x_opt), -1),
                y_for_bo.reshape(len(hil.y_opt), 1))
        new_parameter = hil._get_safe_bo_suggestion(raw)
        hil.x = np.concatenate((
            hil.x, new_parameter.reshape(1, config['Optimization']['n_parms'])
        ), axis=0)

    st.session_state.current_trial += 1
    return True


# ===========================================================================
# Plots
# ===========================================================================

def plot_torque_curve(L0: float, attach: float) -> go.Figure:
    """Plot torque curve for (L0, attach) with R fixed at 0.28m."""
    angles, torques = compute_torque_curve(L0, attach_ratio=attach, angle_min=-30.0, angle_max=30.0)
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
        title=f"Torque-Angle Curve  |  R = {R_FIXED:.2f} m (fixed)  L₀ = {L0:.4f} m  attach = {attach:+.2f}",
        xaxis_title="Ankle Angle (deg)  [DF ← 0 → PF]",
        yaxis_title="Exo Torque (Nm)",
        height=380, margin=dict(l=50, r=50, t=60, b=50),
        showlegend=True, hovermode='x unified',
        xaxis=dict(range=[-30, 30]),
    )
    return fig


def plot_gp_surface():
    """Plot GP surface for (L0, attach) space."""
    import torch
    hil = st.session_state.hil
    if hil.n < st.session_state.config['Optimization']['n_exploration'] + 1:
        return None
    if hil.BO.model is None:
        return None

    config = st.session_state.config
    range_ = np.array(list(config['Optimization']['range'])).reshape(2, 2)
    L0_min, attach_min = range_[0]
    L0_max, attach_max = range_[1]
    n_grid = 40
    L0_grid = np.linspace(L0_min, L0_max, n_grid)
    attach_grid = np.linspace(attach_min, attach_max, n_grid)
    LL, AA = np.meshgrid(L0_grid, attach_grid)
    grid_pts = np.column_stack([LL.ravel(), AA.ravel()])

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
    best_idx = _best_idx(y_obs, config)

    from plotly.subplots import make_subplots as _msp
    fig = _msp(
        rows=1, cols=2,
        subplot_titles=("GP Mean — Predicted Cost", "GP Uncertainty (±1σ)"),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        horizontal_spacing=0.04)

    # --- Panel 1: mean surface ---
    fig.add_trace(go.Surface(
        x=L0_grid, y=attach_grid, z=mean_display,
        colorscale='RdYlGn_r', opacity=0.92,
        colorbar=dict(title='Pred. Cost', x=0.44, len=0.8, thickness=12),
        showscale=True), row=1, col=1)

    if len(x_obs) > 0:
        mask = np.ones(len(x_obs), dtype=bool)
        mask[best_idx] = False
        if mask.any():
            fig.add_trace(go.Scatter3d(
                x=x_obs[mask, 0], y=x_obs[mask, 1], z=y_obs[mask],
                mode='markers', name='Trials',
                marker=dict(size=6, color='#3A2415',
                            line=dict(color='white', width=1))),
                row=1, col=1)
        if _signed_mode(config):
            best_label = (f"Best (SI={y_obs[best_idx]:+.2f}, "
                          f"|Δ|={abs(y_obs[best_idx] - _si_target(config)):.2f})")
        else:
            best_label = f"Best ({y_obs[best_idx]:.2f})"
        fig.add_trace(go.Scatter3d(
            x=[x_obs[best_idx, 0]], y=[x_obs[best_idx, 1]],
            z=[y_obs[best_idx]], mode='markers', name=best_label,
            marker=dict(size=12, color='#E8772E', symbol='diamond',
                        line=dict(color='#7A3A0A', width=2))),
            row=1, col=1)

    # --- Panel 2: uncertainty surface ---
    fig.add_trace(go.Surface(
        x=L0_grid, y=attach_grid, z=std_display,
        colorscale='Oranges',
        colorbar=dict(title='σ', x=1.0, len=0.8, thickness=12),
        showscale=True), row=1, col=2)
    if len(x_obs) > 0:
        fig.add_trace(go.Scatter3d(
            x=x_obs[:, 0], y=x_obs[:, 1], z=np.zeros(len(x_obs)),
            mode='markers', showlegend=False,
            marker=dict(size=5, color='#3A2415', opacity=0.7)),
            row=1, col=2)

    if _signed_mode(config):
        para_str = f"target SI = {_si_target(config):+.1f}%"
    else:
        para_str = "minimize |cost|"
    cam = dict(eye=dict(x=1.5, y=-1.5, z=1.2))
    fig.update_layout(
        title=(f'GP Cost Surface  |  L₀ [{L0_min}–{L0_max}]  '
               f'attach [{attach_min:+.1f}–{attach_max:+.1f}]  |  {para_str}  |  R = {R_FIXED} m (fixed)'),
        height=620, margin=dict(l=0, r=0, t=70, b=0), showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.05),
        scene=dict(xaxis_title='L₀ (m)', yaxis_title='attach',
                   zaxis_title='Pred. Cost', camera=cam),
        scene2=dict(xaxis_title='L₀ (m)', yaxis_title='attach',
                    zaxis_title='σ', camera=cam),
    )
    return fig


def plot_progress():
    if not st.session_state.results:
        return None
    df = pd.DataFrame(st.session_state.results)
    config = st.session_state.config
    signed = _signed_mode(config)
    si_target = _si_target(config)
    baseline_si = st.session_state.baseline_si

    colors = ['gold' if b else 'lightblue' for b in df['is_best']]
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=(
                            f"{'SI' if signed else 'Cost'} vs Trial  "
                            f"(target = {si_target:+.1f}%)" if signed
                            else "Cost vs Trial",
                            'Parameters vs Trial'),
                        vertical_spacing=0.15, row_heights=[0.6, 0.4])
    fig.add_trace(go.Scatter(
        x=df['trial'], y=df['cost'], mode='lines+markers',
        name='SI' if signed else 'Cost',
        marker=dict(size=12, color=colors, line=dict(width=2, color='darkblue')),
        line=dict(color='royalblue', width=2)), row=1, col=1)

    if signed:
        # Target line.
        fig.add_hline(y=si_target, line_dash="dash", line_color="red",
                      annotation_text=f"Target: {si_target:+.1f}%", row=1, col=1)
        # Baseline reference line (where the subject started).
        if baseline_si is not None:
            fig.add_hline(y=baseline_si, line_dash="dot", line_color="gray",
                          annotation_text=f"Baseline: {baseline_si:+.1f}%",
                          annotation_position="top right", row=1, col=1)
        # Best-achieved line.
        best_idx = _best_idx(df['cost'].values, config)
        fig.add_hline(y=df['cost'].iloc[best_idx], line_dash="dot",
                      line_color="goldenrod",
                      annotation_text=f"Best: {df['cost'].iloc[best_idx]:+.2f}%",
                      annotation_position="bottom right",
                      row=1, col=1)
    else:
        fig.add_hline(y=df['cost'].min(), line_dash="dash", line_color="red",
                      annotation_text=f"Best: {df['cost'].min():.2f}",
                      row=1, col=1)

    fig.add_trace(go.Scatter(x=df['trial'], y=df['L0'], mode='lines+markers',
                             name='L₀', marker=dict(size=8),
                             line=dict(width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['trial'], y=df['attach'], mode='lines+markers',
                             name='attach', marker=dict(size=8),
                             line=dict(width=2)), row=2, col=1)
    fig.update_xaxes(title_text="Trial", row=2, col=1)
    fig.update_yaxes(title_text="SI (%)" if signed else "Cost", row=1, col=1)
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

    l_cv = (np.std(np.diff(l_times)) / np.mean(np.diff(l_times))
            if len(l_times) > 1 else np.nan)
    r_cv = (np.std(np.diff(r_times)) / np.mean(np.diff(r_times))
            if len(r_times) > 1 else np.nan)

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

    if trim_seconds > 0:
        for row in (1, 2, 3):
            fig.add_vrect(x0=0, x1=trim_lo_rel, fillcolor='gray', opacity=0.18,
                          layer='below', line_width=0, row=row, col=1)
            fig.add_vrect(x0=trim_hi_rel, x1=max_t, fillcolor='gray', opacity=0.18,
                          layer='below', line_width=0, row=row, col=1)

    if not np.isnan(qc['si_signed']):
        si_text = f"SI = {qc['si_signed']:+.2f}% signed"
        if _signed_mode(st.session_state.config):
            tgt = _si_target(st.session_state.config)
            si_text += f" (target {tgt:+.1f}%, |Δ|={abs(qc['si_signed'] - tgt):.2f}%)"
        subtitle = (f"{si_text}  |  "
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
# ===========================================================================
# ===========================================================================
#  MULTI-PAGE CONSOLE  —  styling, header, navigation, and page functions
# ===========================================================================
# ===========================================================================

import glob

ACCENT = "#E8772E"      # primary orange — matches prelim
ACCENT_DK = "#B5560F"   # deep burnt orange — high-emphasis
ACCENT_DEEP = "#7A3A0A" # darkest orange-brown — header/chips
ACCENT_LT = "#FCA86A"   # light orange — soft fills, hovers
BG = "#FDF4EC"          # light peach-cream background
PANEL = "#FFFFFF"       # white card surface (clean contrast vs bg)
PANEL_2 = "#FBE6D4"     # secondary peach surface
BORDER = "#EFCBA8"      # warm orange-tinted border
TEXT = "#2E1B0E"        # near-black warm brown text
MUTED = "#9B7B63"       # warm muted brown


def inject_theme():
    """Global CSS — refined 'clinical instrument' aesthetic."""
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

      html, body, [class*="css"] {{
          font-family: 'IBM Plex Sans', sans-serif;
          font-weight: 500;
      }}
      h1, h2, h3, h4 {{
          font-family: 'Space Grotesk', sans-serif;
          letter-spacing: -0.02em;
          font-weight: 700 !important;
          color: {TEXT};
      }}
      /* heavier body labels */
      label, .stMarkdown p, p {{ font-weight: 500; }}
      label p {{ font-weight: 600 !important; }}
      .block-container {{
          padding-top: 4rem; padding-bottom: 3rem; max-width: 1800px;
      }}

      /* ---- App header bar — DARK orange anchor ---- */
      .hitlo-header {{
          display: flex; align-items: center; gap: 14px;
          padding: 16px 24px; margin: 0 0 1.6rem 0;
          background: linear-gradient(110deg, {ACCENT_DEEP} 0%, {ACCENT_DK} 55%, {ACCENT} 100%);
          border: none; border-radius: 14px;
          box-shadow: 0 4px 16px rgba(122,58,10,0.28);
      }}
      .hitlo-logo {{
          width: 38px; height: 38px; border-radius: 10px;
          background: rgba(255,255,255,0.18);
          border: 1px solid rgba(255,255,255,0.25);
          display: flex; align-items: center; justify-content: center;
          font-size: 20px;
      }}
      .hitlo-title {{ font-family:'Space Grotesk'; font-weight:700;
          font-size: 1.32rem; color: #FFFFFF; line-height: 1; }}
      .hitlo-sub {{ color: rgba(255,255,255,0.85); font-size: 0.78rem;
          margin-top: 4px; letter-spacing: 0.02em; font-weight: 600; }}
      .hitlo-badge {{ margin-left: auto; padding: 6px 14px; border-radius: 20px;
          background: rgba(255,255,255,0.20); border: 1px solid rgba(255,255,255,0.35);
          color: #FFFFFF; font-size: 0.74rem; font-weight: 700; }}

      /* ---- Metric cards ---- */
      div[data-testid="stMetric"] {{
          background: {PANEL}; border: 1px solid {BORDER};
          border-left: 4px solid {ACCENT}; border-radius: 12px;
          padding: 16px 18px; box-shadow: 0 1px 4px rgba(122,58,10,0.06);
      }}
      div[data-testid="stMetric"] label p {{ color: {MUTED} !important;
          font-size: 0.76rem !important; font-weight: 600 !important;
          text-transform: uppercase; letter-spacing: 0.04em; }}
      div[data-testid="stMetricValue"] {{ font-family:'Space Grotesk';
          font-weight: 700; color: {ACCENT_DK} !important; }}

      /* ---- Buttons ---- */
      .stButton button[kind="primary"] {{
          background: {ACCENT}; color: #2A1505; border: none;
          font-weight: 600; border-radius: 10px; padding: 0.5rem 1rem;
      }}
      .stButton button[kind="primary"]:hover {{ background: #CF6520; }}
      .stButton button[kind="secondary"] {{
          background: {PANEL_2}; color: {TEXT};
          border: 1px solid {BORDER}; border-radius: 10px;
      }}

      /* ---- Expanders & containers ---- */
      div[data-testid="stExpander"] {{
          border: 1px solid {BORDER}; border-radius: 14px;
          background: {PANEL}; overflow: hidden;
      }}
      div[data-testid="stExpander"] summary {{ font-weight: 600; }}
      /* bordered st.container -> white card lifting off peach bg */
      div[data-testid="stVerticalBlockBorderWrapper"] {{
          background: {PANEL}; border: 1px solid {BORDER} !important;
          border-radius: 16px !important;
          box-shadow: 0 2px 10px rgba(122,58,10,0.07);
      }}

      /* ---- Sidebar ---- */
      section[data-testid="stSidebar"] {{
          background: {PANEL}; border-right: 1px solid {BORDER};
      }}
      section[data-testid="stSidebar"] h1,
      section[data-testid="stSidebar"] h2 {{ font-size: 1rem; }}
      /* bigger sidebar nav links */
      [data-testid="stSidebarNav"] a {{
          font-size: 1.02rem !important; font-weight: 600 !important;
          padding: 8px 12px !important;
      }}
      [data-testid="stSidebarNav"] a span {{ font-size: 1.02rem !important; }}

      /* ---- Tables ---- */
      div[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}

      /* ---- Recolor Streamlit's blue accents to orange/white ---- */
      div[data-testid="stAlert"][kind="info"],
      div[data-baseweb="notification"][kind="info"] {{
          background: rgba(232,119,46,0.10) !important;
          border: 1px solid rgba(232,119,46,0.35) !important;
          border-radius: 12px !important;
      }}
      div[data-testid="stAlert"][kind="info"] strong {{
          color: {ACCENT} !important;
      }}
      .stAlert {{ border-radius: 12px !important; }}
      a, a:visited {{ color: {ACCENT} !important; }}
      div[data-testid="stProgress"] div[role="progressbar"] > div {{
          background: {ACCENT} !important;
      }}
      div[data-testid="stSlider"] div[role="slider"] {{
          background: {ACCENT} !important;
      }}
      [data-testid="stSidebarNav"] a[aria-current="page"] {{
          color: {ACCENT} !important;
      }}
      button[data-testid="stNumberInputStepUp"]:hover,
      button[data-testid="stNumberInputStepDown"]:hover {{
          color: {ACCENT} !important; border-color: {ACCENT} !important;
      }}
      div[data-testid="stSpinner"] > div {{ border-top-color: {ACCENT} !important; }}

      /* ---- Section label chip — dark orange ---- */
      .section-chip {{
          display:inline-block; padding: 4px 13px; border-radius: 7px;
          background: {ACCENT_DK}; border:1px solid {ACCENT_DEEP}; color:#FFFFFF;
          font-size: 0.7rem; font-weight:700; letter-spacing:0.07em;
          text-transform: uppercase; margin-bottom: 8px;
      }}
      /* section sub-headers in deep orange for pop */
      .stMarkdown h3, h3 {{ color: {ACCENT_DK} !important; }}
      hr {{ border-color: {BORDER}; }}
    </style>
    """, unsafe_allow_html=True)


def render_header(active: str):
    """Branded top bar with a status badge."""
    if st.session_state.get('initialized'):
        cfg = st.session_state.config
        subj = cfg['Subject']['id'] if cfg else "—"
        badge = f"● {subj} · {active}"
    else:
        badge = f"● {active}"
    st.markdown(f"""
    <div class="hitlo-header">
      <div class="hitlo-logo">🦾</div>
      <div>
        <div class="hitlo-title">HITLO&nbsp;Symmetry</div>
        <div class="hitlo-sub">Human-in-the-Loop Bayesian Optimization · LegExoNET</div>
      </div>
      <div class="hitlo-badge">{badge}</div>
    </div>
    """, unsafe_allow_html=True)


def section_chip(label: str):
    st.markdown(f'<span class="section-chip">{label}</span>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared sidebar status (shown on every page once initialized)
# ---------------------------------------------------------------------------

def sidebar_status():
    if not st.session_state.get('initialized'):
        st.sidebar.info("System not initialized.\nGo to **Setup** to begin.")
        return
    config = st.session_state.config
    opt = config['Optimization']
    range_ = np.array(list(opt['range'])).reshape(2, 2)
    signed = _signed_mode(config)
    si_target = _si_target(config)
    baseline_si = st.session_state.baseline_si

    if _is_aim2(config):
        para = "🩺 Aim 2 — drive to 0"
    elif not st.session_state.baseline_done:
        para = "🧪 Aim 1 — baseline pending"
    elif si_target < 0:
        para = f"🧪 Aim 1 — target {si_target:+.1f}%"
    else:
        para = f"target {si_target:+.1f}%"

    base_line = ("n/a (Aim 2)" if _is_aim2(config)
                 else (f"{baseline_si:+.1f}%" if baseline_si is not None
                       else "pending"))
    st.sidebar.markdown(f"""
    **Session**
    `{config['Subject']['id']} · {config['Subject']['session']}`

    **Paradigm**
    {para}

    **Baseline** {base_line}
    **Trials** {opt['n_steps']} ({opt['n_exploration']} LHS)
    **L₀** {range_[0,0]:.3f}–{range_[1,0]:.3f}  
    **attach** {range_[0,1]:+.1f}–{range_[1,1]:+.1f}
    **R (fixed)** {R_FIXED} m
    """)


# ===========================================================================
# PAGE 1 — SETUP (config editor + initialize)
# ===========================================================================

def page_setup():
    render_header("Setup")
    section_chip("Step 1 · Configure")
    st.header("Experiment Configuration")

    # If already initialized, show status + re-init option.
    if st.session_state.initialized and st.session_state.config_saved:
        st.success("✅ System initialized. Use **Run Experiment** in the nav.")
        sidebar_status()
        if st.button("🔄 Re-initialize / start a new session"):
            st.session_state.initialized = False
            st.session_state.config_saved = False
            st.rerun()
        return

    _cfg = load_config()
    if _cfg is None:
        st.error("No config found. Defaults loaded — edit and save.")
        _cfg = {
            'Subject': {'id': 'P001', 'session': 'S001',
                        'base_dir': '/Users/maccamardo/HITLO_Data'},
            'Cost': {'aim': 'Aim 1', 'sample_rate': 200, 'time': 90,
                     'signed': True, 'si_target': -10.0, 'trim_seconds': 3.0},
            'Optimization': {
                'n_parms': 2, 'n_steps': 15, 'n_exploration': 5,
                'range': [[0.32, -0.2], [0.44, 1.0]],
                'device': 'cpu', 'normalize': True, 'acquisition': 'ei',
                'kernel_function': 'se', 'model_save_path': 'auto',
                'result_save_path': 'auto',
                'max_pf_torque_nm': 90.0, 'pf_check_angle_range': [0.0, 30.0],
                'max_df_torque_nm': 10.0, 'df_check_angle_range': [-30.0, 0.0],
                'slack_at_neutral_max_torque': 2.0, 'df_check_angle_deg': -10.0,
                'lambda_pf': 0.01, 'mu_df': 0.005},
        }
    subj = _cfg.setdefault('Subject', {})
    cost = _cfg.setdefault('Cost', {})
    opt = _cfg.setdefault('Optimization', {})

    with st.container(border=True):
        st.subheader("👤 Subject")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            subj['id'] = st.text_input("Participant ID",
                                       value=str(subj.get('id', 'P001')))
        with cc2:
            subj['session'] = st.text_input(
                "Session", value=str(subj.get('session', 'S001')))
        with cc3:
            subj['base_dir'] = st.text_input(
                "Base data dir",
                value=str(subj.get('base_dir', '/Users/maccamardo/HITLO_Data')))

    with st.container(border=True):
        st.subheader("🎯 Paradigm")
        _aim_options = ["Aim 1 — healthy (induce asymmetry, baseline-relative)",
                        "Aim 2 — stroke (drive to symmetry, target = 0)"]
        _saved_aim = cost.get('aim', 'Aim 1')
        _idx = 1 if str(_saved_aim).startswith('Aim 2') else 0
        aim_choice = st.radio("Experiment aim", _aim_options, index=_idx,
                              horizontal=True)
        is_aim2 = aim_choice.startswith("Aim 2")
        cost['aim'] = "Aim 2" if is_aim2 else "Aim 1"

        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            cost['signed'] = st.checkbox(
                "Signed (two shank sensors)",
                value=bool(cost.get('signed', True)),
                help="Required for both aims.")
        with pc2:
            if is_aim2:
                cost['si_target'] = 0.0
                st.number_input("si_target (%)", value=0.0, disabled=True,
                                help="Aim 2: fixed at 0. No baseline phase.")
            else:
                cost['si_target'] = st.number_input(
                    "Fallback si_target (%)",
                    value=float(cost.get('si_target', -10.0)),
                    min_value=-50.0, max_value=50.0, step=1.0,
                    help="Baseline phase overrides this.")
        with pc3:
            cost['time'] = st.number_input(
                "Trial duration (s)", value=int(cost.get('time', 90)),
                min_value=10, max_value=600, step=5)

        if is_aim2:
            st.info("🩺 **Aim 2 stroke:** baseline skipped, target = 0.")
        else:
            st.info("🧪 **Aim 1 healthy:** baseline phase runs first "
                    "(Pre run-002), target = baseline − displacement.")

        pc4, pc5 = st.columns(2)
        with pc4:
            cost['sample_rate'] = st.number_input(
                "Sample rate (Hz)", value=int(cost.get('sample_rate', 200)),
                min_value=50, max_value=1000, step=10)
        with pc5:
            cost['trim_seconds'] = st.number_input(
                "Trim each end (s)",
                value=float(cost.get('trim_seconds', 3.0)),
                min_value=0.0, max_value=30.0, step=0.5)

    with st.container(border=True):
        st.subheader("🔬 Optimization")
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            opt['n_steps'] = st.number_input(
                "Total trials", value=int(opt.get('n_steps', 15)),
                min_value=1, max_value=100, step=1)
        with oc2:
            opt['n_exploration'] = st.number_input(
                "Exploration (LHS)", value=int(opt.get('n_exploration', 5)),
                min_value=1, max_value=50, step=1)
        with oc3:
            opt['normalize'] = st.checkbox(
                "Normalize", value=bool(opt.get('normalize', True)))

        st.markdown("**Parameter ranges (L₀, attach)** — R is FIXED at 0.28 m")
        rng = np.array(opt.get('range', [[0.32, -0.2], [0.44, 1.0]]),
                       dtype=float)
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            l0_min = st.number_input("L₀ min (m)", value=float(rng[0, 0]),
                                     step=0.01, format="%.4f")
        with rc2:
            l0_max = st.number_input("L₀ max (m)", value=float(rng[1, 0]),
                                     step=0.01, format="%.4f")
        with rc3:
            attach_min = st.number_input("attach min", value=float(rng[0, 1]),
                                         step=0.1, format="%+.2f")
        with rc4:
            attach_max = st.number_input("attach max", value=float(rng[1, 1]),
                                         step=0.1, format="%+.2f")
        opt['range'] = [[l0_min, attach_min], [l0_max, attach_max]]

        with st.expander("⚙️ Advanced torque constraints"):
            st.markdown("**Hard limits enforced every trial:**")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                opt['max_pf_torque_nm'] = st.number_input(
                    "Max PF torque (Nm)",
                    value=float(opt.get('max_pf_torque_nm', 90.0)), step=5.0)
                opt['pf_check_angle_range'] = [
                    st.number_input("PF zone min (deg)", value=0.0, step=1.0),
                    st.number_input("PF zone max (deg)", value=30.0, step=1.0)
                ]
            with sc2:
                opt['max_df_torque_nm'] = st.number_input(
                    "Max DF torque (Nm)",
                    value=float(opt.get('max_df_torque_nm', 10.0)), step=1.0)
                opt['df_check_angle_range'] = [
                    st.number_input("DF zone min (deg)", value=-30.0, step=1.0),
                    st.number_input("DF zone max (deg)", value=0.0, step=1.0)
                ]
            with sc3:
                opt['slack_at_neutral_max_torque'] = st.number_input(
                    "Slack at 0° max (Nm)",
                    value=float(opt.get('slack_at_neutral_max_torque', 2.0)),
                    step=0.5)
                opt['df_check_angle_deg'] = st.number_input(
                    "DF check angle (deg)",
                    value=float(opt.get('df_check_angle_deg', -10.0)), step=1.0)

    st.markdown("---")
    bc1, bc2, bc3 = st.columns([1, 1, 1])
    with bc1:
        if st.button("💾 Save config", type="primary", width="stretch"):
            if save_config_to_disk(_cfg):
                st.session_state.config_saved = True
                st.success("✅ Saved.")
                st.rerun()
    with bc2:
        if st.button("⏭️ Use existing config", width="stretch"):
            st.session_state.config_saved = True
            st.rerun()
    with bc3:
        if st.button("🚀 Initialize system", type="primary", width="stretch",
                     disabled=not st.session_state.config_saved):
            ok, _ = initialize_system(fresh_start=True)
            if ok:
                st.success("✅ Initialized! Go to **Run Experiment**.")
                st.rerun()

    if not st.session_state.config_saved:
        st.caption("Save (or use existing) config to enable Initialize.")

    with st.expander("📄 Preview YAML"):
        st.code(yaml.safe_dump(_cfg, sort_keys=False), language="yaml")


# ===========================================================================
# PAGE 2 — RUN EXPERIMENT (sensors + baseline + trial loop)
# ===========================================================================

def page_run():
    render_header("Run")
    sidebar_status()

    if not st.session_state.initialized:
        st.warning("System not initialized. Go to **Setup** first.")
        return

    config = st.session_state.config
    signed = _signed_mode(config)

    # ---- Live sensors ----
    section_chip("Sensors")
    st.subheader("📡 Live Polar H10 — Left + Right Shank")
    connect_to_lsl()

    @st.fragment(run_every=5.0)
    def _live():
        for side in ['left', 'right']:
            inlet = st.session_state[f'lsl_inlet_{side}']
            if inlet is not None:
                try:
                    inlet.pull_chunk(timeout=0.0, max_samples=1)
                except Exception:
                    st.session_state[f'lsl_inlet_{side}'] = None
        connect_to_lsl()
        sr = st.session_state.config['Cost']['sample_rate']
        cL, cR = st.columns(2)
        for side, col in [('left', cL), ('right', cR)]:
            inlet = st.session_state[f'lsl_inlet_{side}']
            store = st.session_state[f'live_data_{side}']
            with col:
                if inlet is not None:
                    update_live_data(inlet, store)
                    st.success(f"✅ {side.capitalize()} · "
                               f"{len(store['time'])} samples")
                    fig = plot_live_sensor(store, f'polar accel {side}', sr)
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info(f"⏳ Collecting {side}…")
                else:
                    st.warning(f"⚠️ {side.capitalize()} not found")
                    if st.button(f"🔄 Reconnect {side}", key=f"rc_{side}"):
                        st.session_state[f'lsl_inlet_{side}'] = None
                        st.session_state[f'live_data_{side}'] = {
                            'time': [], 'x': [], 'y': [], 'z': []}
                        connect_to_lsl()
                        st.rerun()
    _live()

    missing = []
    if st.session_state.lsl_inlet_left is None:
        missing.append('LEFT')
    if st.session_state.lsl_inlet_right is None:
        missing.append('RIGHT')
    if missing:
        st.error(f"🚨 SENSOR DISCONNECTED — {', '.join(missing)}. "
                 f"Do NOT start a trial.")
    st.markdown("---")

    # ---- Aim 2: force target 0, skip baseline ----
    if _is_aim2(config):
        if not st.session_state.baseline_done:
            config['Cost']['si_target'] = 0.0
            st.session_state.cost_extractor.si_target = 0.0
            st.session_state.hil.si_target = 0.0
            st.session_state.hil._bo_direction_str = "|SI - 0.0|"
            st.session_state.baseline_si = None
            st.session_state.baseline_displacement = None
            st.session_state.baseline_done = True
            save_checkpoint()

    # ---- Baseline phase (Aim 1) ----
    if signed and not _is_aim2(config) and not st.session_state.baseline_done:
        _baseline_phase(config)
        return

    # ---- Optimization phase ----
    _optimization_phase(config, signed)


def _baseline_phase(config):
    section_chip("Step 2 · Baseline")
    st.header("📏 Baseline Measurement")
    st.info(
        f"**Pre run-{BASELINE_IGNORE_RUN:03d}** — familiarization, **ignored**. "
        f"**Pre run-{BASELINE_TRIAL_RUN:03d}** — THE baseline trial. "
        f"Target = baseline − displacement.")

    base_fn = baseline_filename(BASELINE_TRIAL_RUN)
    ignore_fn = baseline_filename(BASELINE_IGNORE_RUN)

    with st.expander("📝 Instructions", expanded=True):
        st.markdown(f"""
        **1. Familiarization (ignored):** no device · Block/Task = `{BASELINE_TASK}` ·
        Run = `{BASELINE_IGNORE_RUN}` → `{ignore_fn}`

        **2. Baseline trial:** band slack · Block/Task = `{BASELINE_TASK}` ·
        Run = `{BASELINE_TRIAL_RUN}` → `{base_fn}` ·
        save to `{st.session_state.cost_extractor.trial_data_dir}`
        · walk {config['Cost']['time']} s.

        **3. Analyze run-002 below.**
        """)

    if st.session_state.baseline_si is not None:
        st.success(f"Baseline SI (run-{BASELINE_TRIAL_RUN:03d}): "
                   f"**{st.session_state.baseline_si:+.2f}%**")

    cb1, cb2, cb3 = st.columns(3)
    with cb1:
        disp = st.number_input("Displacement (%)", value=10.0, min_value=1.0,
                               max_value=30.0, step=1.0, key="bl_disp")
    with cb2:
        ready = baseline_file_exists(BASELINE_TRIAL_RUN)
        if ready:
            st.success(f"✅ Found {base_fn}")
        else:
            st.warning(f"⏳ Waiting for {base_fn}")
        if st.button("📊 Analyze baseline (run-002)", type="primary",
                     width="stretch", disabled=not ready):
            si = analyze_baseline(BASELINE_TRIAL_RUN)
            if si is None:
                st.error("Analysis failed.")
            else:
                st.session_state.baseline_si = si
                st.rerun()
    with cb3:
        locked = (st.session_state.baseline_si is None)
        if st.button("✅ Lock target & start", width="stretch",
                     disabled=locked):
            base = float(st.session_state.baseline_si)
            target = compute_baseline_target(base, float(disp))
            st.session_state.baseline_displacement = float(disp)
            config['Cost']['si_target'] = target
            st.session_state.cost_extractor.si_target = target
            st.session_state.hil.si_target = target
            st.session_state.hil._bo_direction_str = f"|SI - {target:+.1f}|"
            st.session_state.baseline_done = True
            save_checkpoint()
            st.rerun()

    if st.session_state.baseline_si is not None:
        pv = compute_baseline_target(float(st.session_state.baseline_si),
                                     float(disp))
        st.caption(f"→ target = {st.session_state.baseline_si:+.2f}% "
                   f"− {disp:.0f}% = **{pv:+.2f}%**")


def _optimization_phase(config, signed):
    hil = st.session_state.hil
    trial_num = st.session_state.current_trial + 1
    n_steps = config['Optimization']['n_steps']
    n_exploration = config['Optimization']['n_exploration']
    si_target = _si_target(config)

    if trial_num > n_steps:
        st.success("🎉 OPTIMIZATION COMPLETE — see **Results**.")
        return

    section_chip(f"Step 3 · Trial {trial_num} of {n_steps}")
    c1, c2 = st.columns([2, 1])
    with c1:
        if trial_num <= n_exploration:
            st.subheader(f"🎲 Exploration {trial_num}/{n_exploration}")
        else:
            st.subheader(f"🧠 Bayesian Optimization "
                         f"{trial_num - n_exploration}/{n_steps - n_exploration}")
    with c2:
        st.progress((trial_num - 1) / n_steps,
                    text=f"{int((trial_num-1)/n_steps*100)}%")

    # Generate next BO suggestion if needed
    if hil.n >= len(hil.x):
        if len(hil.x_opt) >= n_exploration:
            try:
                if config['Optimization']['normalize']:
                    nx = hil._normalize_x(hil.x_opt)
                    ny = hil._mean_normalize_y(hil.y_opt)
                    raw = hil.BO.run(nx.reshape(len(hil.x_opt), -1),
                                     ny.reshape(len(hil.x_opt), 1))
                    raw = hil._denormalize_x(raw)
                else:
                    yb = (-np.abs(hil.y_opt - si_target) if signed
                          else -hil.y_opt)
                    raw = hil.BO.run(hil.x_opt.reshape(len(hil.x_opt), -1),
                                     yb.reshape(len(hil.y_opt), -1))
                newp = hil._get_safe_bo_suggestion(raw)
                hil.x = np.concatenate((hil.x, newp.reshape(
                    1, config['Optimization']['n_parms'])), axis=0)
            except Exception as e:
                st.error(f"BO suggestion failed: {e}")
                return
        else:
            st.error("Parameter index out of bounds — reinitialize.")
            return

    params = hil.x[hil.n]

    # NEW: Show L0 and attach (not R and L0)
    mc1, mc2 = st.columns(2)
    mc1.metric("L₀ (m)", f"{params[0]:.4f}")
    mc2.metric("Attach", f"{params[1]:+.2f}")

    # NEW: R is fixed at 0.28m
    st.plotly_chart(plot_torque_curve(L0=params[0], attach=params[1]),
                    width="stretch")

    with st.expander("📝 LabRecorder steps", expanded=True):
        st.markdown(f"""
        1. Enter **L₀ = {params[0]:.4f}** and **attach = {params[1]:+.2f}** into Computer 2.
           (R is fixed at {R_FIXED} m)
        2. LabRecorder: Block/Task = `Default`, Run = `{trial_num}` →
           `sub-{config['Subject']['id']}_ses-{config['Subject']['session']}_task-Default_run-{trial_num:03d}_eeg.xdf`
        3. Walk {config['Cost']['time']} s · Stop · Analyze below.
        """)

    a1, a2, a3 = st.columns(3)
    with a1:
        ok = check_file_exists(trial_num)
        if ok:
            st.success(f"✅ run-{trial_num:03d}.xdf")
        else:
            st.warning(f"⏳ run-{trial_num:03d}.xdf")
    with a2:
        if st.button("🔍 Check file", width="stretch"):
            st.rerun()
    with a3:
        if st.button("▶️ Analyze Trial", type="primary", width="stretch",
                     disabled=not ok):
            with st.spinner("Analyzing…"):
                if analyze_current_trial():
                    st.rerun()
                else:
                    st.error("Analysis failed.")

    # Latest QC inline
    if st.session_state.results:
        st.markdown("---")
        last = st.session_state.results[-1]['trial']
        fn = trial_filename(config['Subject']['id'],
                            config['Subject']['session'], last)
        fp = os.path.join(st.session_state.cost_extractor.trial_data_dir, fn)
        trim_s = config['Cost'].get('trim_seconds', 3.0)
        cfg = st.session_state.cost_extractor.detection_cfg
        res = plot_heelstrikes_last_trial(fp, cfg, trim_seconds=trim_s)
        if res is not None:
            hs_fig, hs_warn = res
            with st.expander(f"🦶 Heel-strike QC — trial {last}", expanded=True):
                for w in hs_warn:
                    st.error(w) if "🚨" in w else st.warning(w)
                if not hs_warn:
                    st.success("✅ QC clean.")
                if hs_fig is not None:
                    st.plotly_chart(hs_fig, width="stretch")


# ===========================================================================
# PAGE 3 — GP VIEWER (live + load past session)
# ===========================================================================

def page_gp_viewer():
    render_header("GP Viewer")
    sidebar_status()
    section_chip("Model Inspection")
    st.header("🧠 GP Optimization Viewer")

    mode = st.radio("Source", ["Live (current session)",
                               "Load past session"],
                    horizontal=True)

    if mode.startswith("Live"):
        if not st.session_state.initialized:
            st.warning("No live session. Initialize in **Setup**, or switch "
                       "to **Load past session**.")
            return
        hil = st.session_state.hil
        n_exp = st.session_state.config['Optimization']['n_exploration']
        if hil.n < n_exp + 1 or hil.BO.model is None:
            st.info(f"Live GP appears after exploration "
                    f"({n_exp} trials) + 1 BO trial. "
                    f"Currently {hil.n} trials done.")
            return
        fig = plot_gp_surface()
        if fig:
            st.plotly_chart(fig, width="stretch")
        if st.session_state.results:
            st.markdown("---")
            section_chip("Trial history")
            df = pd.DataFrame(st.session_state.results)
            st.dataframe(df[['trial', 'L0', 'attach', 'cost', 'phase']],
                         width="stretch")
        return

    # ---- Load past session ----
    base_default = (st.session_state.config['Subject']['base_dir']
                    if st.session_state.get('config')
                    else '/Users/maccamardo/HITLO_Data')
    c1, c2, c3 = st.columns(3)
    with c1:
        base_dir = st.text_input("Base dir", value=base_default)
    with c2:
        subject = st.text_input("Subject", value="P048")
    with c3:
        session = st.text_input("Session", value="S001")

    _gp_historical_viewer(base_dir, subject, session)


def _gp_historical_viewer(base_dir, subject, session):
    """Load saved GP checkpoints from disk and scrub iterations."""
    import torch
    from botorch.models import SingleTaskGP
    from gpytorch.likelihoods import GaussianLikelihood

    models_dir = Path(f'{base_dir}/sub-{subject}/ses-{session}/'
                      f'derivatives/hil_optimization/models')
    hil_csv = Path(f'{base_dir}/sub-{subject}/ses-{session}/eeg/'
                   f'sub-{subject}_ses-{session}_hil_results.csv')

    if not hil_csv.exists():
        st.error(f"No results at {hil_csv}")
        return
    if not models_dir.exists():
        st.error(f"No models dir at {models_dir}")
        return

    hil_results = pd.read_csv(hil_csv)
    iter_folders = sorted(
        [f for f in models_dir.glob('iter_*') if f.is_dir()],
        key=lambda x: int(x.name.split('_')[1]))
    if not iter_folders:
        st.error("No iteration checkpoints found.")
        return
    iters = [int(f.name.split('_')[1]) for f in iter_folders]

    cfg = load_config()
    rng = (np.array(list(cfg['Optimization']['range'])).reshape(2, 2)
           if cfg else np.array([[0.32, -0.2], [0.44, 1.0]]))
    L0_MIN, attach_MIN = rng[0, 0], rng[0, 1]
    L0_MAX, attach_MAX = rng[1, 0], rng[1, 1]

    st.success(f"✅ {len(hil_results)} trials · {len(iters)} checkpoints "
               f"(iter {min(iters)}–{max(iters)})")
    sel = st.select_slider("Iteration", options=iters, value=iters[0])

    ipath = models_dir / f'iter_{sel}'
    try:
        data = np.loadtxt(ipath / 'data.csv')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        L0n, attachn, bo = data[:, 0], data[:, 1], data[:, 2]
        L0p = L0n * (L0_MAX - L0_MIN) + L0_MIN
        attachp = attachn * (attach_MAX - attach_MIN) + attach_MIN
        actual = hil_results['cost'].values[:sel]

        L0g = np.linspace(L0_MIN, L0_MAX, 40)
        attachg = np.linspace(attach_MIN, attach_MAX, 40)
        LL, AA = np.meshgrid(L0g, attachg)
        Xt = np.column_stack([((LL - L0_MIN)/(L0_MAX - L0_MIN)).ravel(),
                              ((AA - attach_MIN)/(attach_MAX - attach_MIN)).ravel()])
        Xtr = torch.tensor(np.column_stack([L0n, attachn]), dtype=torch.float64)
        ytr = torch.tensor(bo, dtype=torch.float64).reshape(-1, 1)
        lik = GaussianLikelihood()
        model = SingleTaskGP(Xtr, ytr, likelihood=lik)
        model.load_state_dict(torch.load(ipath / 'model.pth'), strict=False)
        model.eval()
        with torch.no_grad():
            pred = lik(model(torch.tensor(Xt, dtype=torch.float64)))
            mean = pred.mean.numpy().reshape(LL.shape)
            lo, up = pred.confidence_region()
            unc = (up.numpy() - lo.numpy()).reshape(LL.shape)
    except Exception as e:
        st.error(f"Could not load iteration {sel}: {e}")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Iteration", sel)
    m2.metric("Trials so far", len(L0p))
    m3.metric("|Best SI|", f"{np.abs(actual).min():.2f}%"
              if len(actual) else "—")

    from plotly.subplots import make_subplots as _msp
    fig = _msp(rows=1, cols=2,
               subplot_titles=("GP mean (BO score)", "GP uncertainty"),
               specs=[[{'type': 'surface'}, {'type': 'surface'}]],
               horizontal_spacing=0.05)
    fig.add_trace(go.Surface(x=L0g, y=attachg, z=mean, colorscale='RdYlGn',
                  colorbar=dict(x=0.43, len=0.8, thickness=12)), row=1, col=1)
    _ns = min(len(L0p), len(attachp), len(bo))
    fig.add_trace(go.Scatter3d(
        x=L0p[:_ns], y=attachp[:_ns], z=bo[:_ns], mode='markers+text',
        marker=dict(size=6, color='red', line=dict(color='black', width=1)),
        text=[str(i+1) for i in range(_ns)],
        textfont=dict(size=9, color='white'), name='trials'), row=1, col=1)
    fig.add_trace(go.Surface(x=L0g, y=attachg, z=unc, colorscale='Reds',
                  colorbar=dict(x=1.0, len=0.8, thickness=12)), row=1, col=2)
    cam = dict(eye=dict(x=1.5, y=1.5, z=1.3))
    fig.update_layout(height=560, showlegend=False,
                      scene=dict(xaxis_title='L₀', yaxis_title='attach',
                                 zaxis_title='BO', camera=cam),
                      scene2=dict(xaxis_title='L₀', yaxis_title='attach',
                                  zaxis_title='σ', camera=cam),
                      margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")

    with st.expander("📊 Trial data"):
        n = min(len(L0p), len(attachp), len(bo), len(actual))
        if n == 0:
            st.info("No overlapping trial data to display for this iteration.")
        else:
            st.dataframe(pd.DataFrame({
                'Trial': range(1, n + 1),
                'L₀': L0p[:n], 'attach': attachp[:n],
                'BO cost': bo[:n],
                'signed SI %': actual[:n]}), width="stretch")


# ===========================================================================
# PAGE 4 — RESULTS
# ===========================================================================

def page_results():
    render_header("Results")
    sidebar_status()
    section_chip("Outcomes")
    st.header("📊 Results & History")

    if not st.session_state.initialized or not st.session_state.results:
        st.info("No results yet. Run trials in **Run Experiment**.")
        return

    config = st.session_state.config
    signed = _signed_mode(config)
    si_target = _si_target(config)
    hil = st.session_state.hil
    n_steps = config['Optimization']['n_steps']

    best_idx = _best_idx(np.array([r['cost'] for r in st.session_state.results]),
                         config)
    best_cost = st.session_state.results[best_idx]['cost']
    latest = st.session_state.results[-1]['cost']
    done = len(st.session_state.results)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Trials", f"{done}/{n_steps}")
    with m2:
        if signed:
            st.metric(f"Best SI (→{si_target:+.0f}%)", f"{best_cost:+.2f}%",
                      delta=f"|Δ| {abs(best_cost-si_target):.2f}",
                      delta_color="off")
        else:
            st.metric("Best cost", f"{best_cost:.3f}")
    with m3:
        st.metric("Latest SI" if signed else "Latest", f"{latest:+.2f}%"
                  if signed else f"{latest:.3f}")
    with m4:
        if st.session_state.baseline_si is not None:
            st.metric("Baseline", f"{st.session_state.baseline_si:+.2f}%")
        else:
            st.metric("Baseline", "n/a")

    if (fig := plot_progress()):
        st.plotly_chart(fig, width="stretch")

    if len(st.session_state.results) > config['Optimization']['n_exploration']:
        section_chip("GP surface")
        if (gp := plot_gp_surface()):
            st.plotly_chart(gp, width="stretch")

    section_chip("Trial history")
    df = pd.DataFrame(st.session_state.results)
    disp = df.copy()
    disp['L0'] = disp['L0'].map('{:.4f}'.format)
    disp['attach'] = disp['attach'].map('{:+.2f}'.format)
    disp['cost'] = disp['cost'].map(
        '{:+.4f}'.format if signed else '{:.4f}'.format)
    if 'dist_from_target' in disp.columns:
        disp['|Δ target|'] = disp['dist_from_target'].map('{:.4f}'.format)
    disp['best'] = disp['is_best'].map(lambda x: '⭐' if x else '')
    cols = ['trial', 'L0', 'attach', 'cost']
    if '|Δ target|' in disp.columns and signed:
        cols.append('|Δ target|')
    cols += ['phase', 'best']
    st.dataframe(disp[cols], width="stretch", height=360)

    csv = df.to_csv(index=False)
    st.download_button("💾 Download results CSV", data=csv,
                       file_name=f"{config['Subject']['id']}_results.csv",
                       mime="text/csv", type="primary")


# ===========================================================================
# NAVIGATION ENTRY POINT
# ===========================================================================

st.set_page_config(page_title="HITLO Symmetry Console",
                   page_icon="🦾", layout="wide")
inject_theme()

_pages = [
    st.Page(page_setup, title="Setup", icon="⚙️"),
    st.Page(page_run, title="Run Experiment", icon="🏃"),
    st.Page(page_gp_viewer, title="GP Viewer", icon="🧠"),
    st.Page(page_results, title="Results", icon="📊"),
]
nav = st.navigation(_pages, position="sidebar")
nav.run()