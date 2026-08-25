"""
apps/hitlo_console.py — HITLO_Symmetry multi-page console.

Version 3.0.0 — unified index parameterization.

BO optimizes ONE number, x in [-1, +1], resolved through index_unified.csv into
the four physical parameters the operator sets: (R, theta, L0, attachment_ratio).
The console shows x for context but leads with the physical values, since x is
meaningless at the bench.

This is the clinician-facing tool. It:
  - Shows live Polar H10 streams so you can confirm sensors before each trial
  - Runs a BASELINE phase first (no-band "Pre" trials) to measure the
    subject's natural asymmetry, then computes the optimization target
    relative to that baseline
  - Displays the four physical parameters the BO wants you to set this trial
  - Shows where that trial sits on the index (dose and stiffness)
  - After each trial, analyzes the XDF and shows a QC plot of heel strikes
  - Tracks progress, cost, and the GP posterior across trials
  - Auto-checkpoints so a crash doesn't lose your session

WHAT CHANGED FROM 2.x
The torque-curve plot is gone. It recomputed torque from a Python forward model
that has since been deleted: that model ran at a different spring rate with R and
theta hardcoded, so it described a device the index table does not build. The
table already carries validated stiffness and dose per row, so the console plots
those instead. The GP surface is now a 1-D posterior rather than a pair of 3-D
surfaces over (L0, attach). Safety limits are no longer editable here — they live
in build_index_unified.m, and changing them means regenerating the CSV.

PARADIGM (Cost.si_target in config, OR set live from baseline)
--------------------------------------------------------------
  - si_target =   0.0 → drive SI toward 0 (Aim 2 stroke, "minimize asymmetry")
  - si_target < 0      → drive SI toward a negative target (Aim 1 healthy,
                         "induce target asymmetry" via passive band)

CONFIG EDITOR (first page)
--------------------------
On launch, before initialization, the operator edits the experiment config
(subject, session, target paradigm, ramp, etc.) directly in the UI and saves it
back to config/exo_symmetry_config.yml. No hand-editing the YAML.

BASELINE-RELATIVE TARGETING (Aim 1)
-----------------------------------
For healthy subjects, the goal is to induce a fixed *displacement* from the
subject's own baseline, not a fixed absolute SI. The baseline phase:
  1. Pre run-001 = no-device familiarization trial — IGNORED (not analyzed).
  2. Pre run-002 = THE baseline trial (band slack / no perturbation).
     Its signed SI alone defines baseline_si (no averaging).
  3. Target is set to:  si_target = baseline_si + sign(baseline_si) × displacement
     (amplifies the subject's existing asymmetry in its own direction; inside
      a ±1.5% deadband the sign isn't resolvable, so the device's default
      direction is used)
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
import time
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
from hitlo.cost import SymmetryCost
from hitlo.detection import detect_heelstrikes_full, DetectionConfig
from hitlo.detectors import detect as detect_strikes
from hitlo.io import (load_both_polar_streams, load_streams,
                      trial_filename, backend_modality)
from hitlo.symmetry import (
    compute_step_times, compute_symmetry_index, trim_peaks,
)


# Baseline trials use this BIDS task tag (LabRecorder Block/Task dropdown).
BASELINE_TASK = "Pre"
# Pre run-001 is a no-device familiarization trial that we IGNORE.
# Pre run-002 is THE baseline trial whose SI defines the target.
BASELINE_IGNORE_RUN = 1
BASELINE_TRIAL_RUN = 2

# Collins/Wiggin/Sawicki 2015 metabolically-optimal ankle stiffness, drawn on
# the index plot as a literature reference.
COLLINS_NM_PER_RAD = 180.0


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


def _direction_label(direction: int) -> str:
    return 'PF' if direction > 0 else 'DF' if direction < 0 else 'ZERO'


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
            # Which table produced these x values. If the CSV is regenerated
            # mid-study the same x means a different configuration, so a
            # checkpoint from the old table is not resumable against the new one.
            'index_csv': str(hil.table.path),
            'n_levels': len(hil.table),
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

def _active_config() -> dict:
    """The config in effect, whether or not the session is initialized yet.

    st.session_state.config stays None until Initialize runs on the Setup
    page. The Sensors page is Step 0 and comes BEFORE that, so relying on
    session state alone made it always fall back to Polar -- it could never
    show the Trigno page however the config was set. Fall back to the file on
    disk, which is what Setup would have loaded anyway.
    """
    cfg = st.session_state.get('config')
    if cfg:
        return cfg
    try:
        import yaml
        with open(_config_path()) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _backend() -> str:
    return (_active_config().get('Sensing') or {}).get('backend', 'polar')


def connect_to_lsl() -> bool:
    """Attach live inlets for whichever backend the config selects.

    The two backends differ structurally, not just in stream name:

      polar    TWO streams, one per side, resolved by name
      trigno   ONE stream carrying both sides, demultiplexed by channel label

    For Trigno both session_state inlets point at the SAME inlet object and
    the side split happens per-sample in update_live_data, using column
    indices worked out once at connect time. Pulling the same inlet twice
    would race — each pull consumes samples the other will never see.
    """
    if (st.session_state.lsl_inlet_left is not None and
            st.session_state.lsl_inlet_right is not None):
        return True
    try:
        from pylsl import StreamInlet, resolve_streams
        # 3s, not pylsl's 1s default. Multicast resolution on a shared lab
        # network is lossy: with several other experiments streaming, a 1s
        # window intermittently fails to hear outlets that are perfectly
        # healthy, and the caller then reports SENSOR DISCONNECTED for sensors
        # that never dropped a sample.
        streams = resolve_streams(wait_time=3.0)

        if _backend() == 'trigno':
            want = (_active_config().get('Sensing') or {}).get(
                'stream', 'TrignoIMU')
            for s in streams:
                if s.name() != want:
                    continue
                inlet = StreamInlet(s)
                cols = _trigno_columns(inlet)
                if cols is None:
                    # No usable labels. Refuse rather than guess a column
                    # order: guessing wrong swaps the legs, which inverts the
                    # sign of the symmetry index while looking plausible.
                    st.session_state.trigno_label_error = (
                        f"'{want}' declares no usable channel labels. Expected "
                        f"left_acc_x ... right_gyr_z. Cannot split left from "
                        f"right without them.")
                    return False
                st.session_state.trigno_cols = cols
                st.session_state.trigno_label_error = None
                st.session_state.lsl_inlet_left = inlet
                st.session_state.lsl_inlet_right = inlet
                break
        else:
            for s in streams:
                if s.name() == 'polar accel left' and st.session_state.lsl_inlet_left is None:
                    st.session_state.lsl_inlet_left = StreamInlet(s)
                if s.name() == 'polar accel right' and st.session_state.lsl_inlet_right is None:
                    st.session_state.lsl_inlet_right = StreamInlet(s)
    except Exception:
        pass
    return (st.session_state.lsl_inlet_left is not None and
            st.session_state.lsl_inlet_right is not None)


def _trigno_columns(inlet):
    """Column indices for each side's accel in a multiplexed Trigno stream.

    Returns {'left': [x,y,z], 'right': [x,y,z]} or None if the labels are
    absent or incomplete. Accel only — the live plot shows acceleration for
    both backends so the display stays comparable.

    LSL SUBTLETY: resolve_streams() returns metadata-LIGHT StreamInfo objects
    with an empty desc, so channel labels are NOT available there. The full
    description only arrives after opening an inlet and calling inlet.info().
    Reading labels off the resolved info silently yields none, which looks
    exactly like a bridge that forgot to declare them.
    """
    try:
        full = inlet.info(timeout=5.0)
        chans = full.desc().child('channels').child('channel')
        labels = []
        while not chans.empty():
            labels.append(chans.child_value('label').strip().lower())
            chans = chans.next_sibling()
    except Exception:
        return None
    if not labels:
        return None
    out = {}
    for side in ('left', 'right'):
        cols = {}
        for i, lab in enumerate(labels):
            if not lab.startswith(side):
                continue
            # SHANK only. A foot or thigh sensor also starts with the side and
            # contains 'acc', and with channel order left_shank, right_foot,
            # right_shank the foot columns come FIRST and would silently be
            # taken as "right" — showing the wrong body segment in the live
            # display and in the leg check.
            if 'foot' in lab or 'thigh' in lab:
                continue
            if not any(tok in lab for tok in ('acc', 'accel')):
                continue
            for ax in ('x', 'y', 'z'):
                if lab.endswith(ax) and ax not in cols:
                    cols[ax] = i
        if len(cols) != 3:
            return None
        out[side] = [cols['x'], cols['y'], cols['z']]
    return out


def update_live_data(inlet, store, side: str = None) -> None:
    """Pull live samples into a display buffer.

    On Trigno both sides share ONE inlet, so a single pull has to feed both
    buffers — pulling twice would race, with each call consuming samples the
    other never sees. When `side` is given and columns are known, the caller
    is on the shared-inlet path and this fills both stores from one pull.
    """
    if inlet is None:
        return
    cols = st.session_state.get('trigno_cols')
    try:
        if cols and _backend() == 'trigno':
            if side not in (None, 'left'):
                return          # the 'left' call already filled both
            samples, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=100)
            if not samples:
                return
            for sd in ('left', 'right'):
                store_sd = st.session_state[f'live_data_{sd}']
                cx, cy, cz = cols[sd]
                for i, sample in enumerate(samples):
                    store_sd['time'].append(timestamps[i])
                    store_sd['x'].append(sample[cx])
                    store_sd['y'].append(sample[cy])
                    store_sd['z'].append(sample[cz])
                mx = st.session_state.max_live_points
                for key in store_sd:
                    if len(store_sd[key]) > mx:
                        store_sd[key] = store_sd[key][-mx:]
            return

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
    # 'eeg' for Polar (an artefact of the original LabRecorder template),
    # 'motion' for Trigno, which is what BIDS specifies for IMU data.
    eeg_dir = os.path.join(base_dir, f"sub-{subject}", f"ses-{session}",
                           backend_modality(config))

    deriv_base = os.path.join(base_dir, f"sub-{subject}", f"ses-{session}",
                              "derivatives", "hil_optimization")
    config['Optimization']['model_save_path'] = os.path.join(deriv_base, "models") + "/"
    config['Optimization']['result_save_path'] = os.path.join(deriv_base, "results") + "/"
    os.makedirs(config['Optimization']['model_save_path'], exist_ok=True)
    os.makedirs(config['Optimization']['result_save_path'], exist_ok=True)
    os.makedirs(eeg_dir, exist_ok=True)

    signed = _signed_mode(config)
    si_target = _si_target(config)
    trim_s = config['Cost'].get('trim_seconds', 3.0)

    st.session_state.cost_extractor = SymmetryCost(
        trial_data_dir=eeg_dir,
        subject_id=subject,
        session=session,
        signed=signed,
        si_target=si_target,
        trim_seconds=trim_s,
    )

    # HIL_Exo loads and validates the index table. A missing or malformed CSV
    # raises here, at setup, rather than mid-session with a subject waiting.
    try:
        st.session_state.hil = HIL_Exo(
            st.session_state.config, st.session_state.cost_extractor)
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Could not load the index table: {e}")
        return False, False

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
                f"Checkpoint paradigm mismatch: checkpoint had "
                f"signed={ckpt_signed}; config has signed={signed}. "
                f"Refusing to resume — use Fresh Start, or revert the "
                f"config to match the checkpoint."
            )
            return False, False

        # Detect an index-table change. x values are only meaningful against
        # the table that produced them: regenerate the CSV with different
        # level counts or a different band ceiling and x = -1 now points at a
        # different configuration. Resuming across that would silently mix
        # two search spaces in one dataset.
        ckpt_levels = ckpt.get('n_levels', None)
        if ckpt_levels is not None and ckpt_levels != len(st.session_state.hil.table):
            st.warning(
                f"Index table mismatch: checkpoint ran against {ckpt_levels} "
                f"levels; the current table has {len(st.session_state.hil.table)}. "
                f"The same x means a different configuration across tables. "
                f"Refusing to resume — use Fresh Start, or restore the CSV "
                f"this session was run with."
            )
            return False, False

        try:
            hil = st.session_state.hil
            hil.x = np.array(ckpt['x']).reshape(-1, 1)
            hil.x_opt = (np.array(ckpt['x_opt']).reshape(-1, 1)
                         if ckpt['x_opt'] else np.array([]))
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
            st.warning(f"Checkpoint found but could not load ({e}). Starting fresh.")
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
                          run_num, task=BASELINE_TASK,
                          modality=backend_modality(config))


def baseline_file_exists(run_num: int) -> bool:
    fp = os.path.join(st.session_state.cost_extractor.trial_data_dir,
                      baseline_filename(run_num))
    return os.path.exists(fp)


def analyze_baseline(run_num: int) -> Optional[float]:
    """Analyze a no-band baseline trial; return its signed SI.

    Uses the SAME detection + symmetry pipeline as the optimization cost so
    the baseline SI is directly comparable to the trial SIs. The exoskeleton
    configuration is irrelevant here (band slack), and the cost function no
    longer knows about the configuration at all.
    """
    fname = baseline_filename(run_num)
    fp = os.path.join(st.session_state.cost_extractor.trial_data_dir, fname)
    if not os.path.exists(fp):
        return None
    analysis = st.session_state.cost_extractor.analyze_trial(
        trial_num=run_num, filename=fname, verbose=False)
    if analysis is None:
        return None
    return float(analysis.symmetry_index)


# Below this |baseline SI|, the subject has no meaningful directional
# asymmetry and sign(baseline) is a coin flip inside measurement noise.
# Amplifying a sign you can't resolve would assign direction at random
# across subjects, so these fall back to the device's default direction.
BASELINE_DEADBAND_PCT = 1.5
DEFAULT_DIRECTION = -1.0   # negative = longer left step (left-side device)


def compute_baseline_target(baseline_si: float, displacement: float) -> float:
    """Optimization target = the subject's own asymmetry, amplified.

    The goal for Aim 1 is error augmentation: take whatever asymmetry the
    subject already walks with and make it LARGER in the same direction, by a
    fixed displacement. The induced displacement is the dose, held constant
    across subjects; the direction belongs to the subject.

        baseline -6%  -> target -16%   (already left-dominant, push further)
        baseline +3%  -> target +13%   (already right-dominant, push further)

    This replaces the earlier always-negative rule (target = baseline -
    displacement), which was written when the device could only perturb one
    way. That rule amplified negative-baseline subjects correctly but pushed
    positive-baseline subjects THROUGH zero and out the other side: +3% became
    -7%, which reverses their asymmetry rather than augmenting it, and makes
    the delivered dose 10% for one subject and 3% for another.

    Near zero, sign(baseline) is not resolvable — a subject at +0.2% is not
    meaningfully right-dominant, and taking that sign at face value would hand
    out directions at random. Inside the deadband we use DEFAULT_DIRECTION
    instead, matching the left-side device geometry.

    ASSUMPTION, NOT YET VERIFIED: that the device can drive SI in both
    directions. The unified index spans dorsiflexor resistance through zero to
    plantarflexor assistance, so bidirectional torque is available — but the
    device is unilateral, and it is possible that both polarities perturb the
    left limb the same way and move SI only one direction with different
    magnitudes. If pilot data shows SI never crosses zero regardless of x, a
    positive-baseline subject cannot be amplified and this rule cannot be met
    for them.
    """
    if abs(baseline_si) < BASELINE_DEADBAND_PCT:
        direction = DEFAULT_DIRECTION
    else:
        direction = 1.0 if baseline_si > 0 else -1.0
    return baseline_si + direction * displacement

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
        modality=backend_modality(st.session_state.config),
    )
    return os.path.exists(os.path.join(trial_dir, fname))


def analyze_current_trial() -> bool:
    trial_num = st.session_state.current_trial + 1
    config = st.session_state.config
    fname = trial_filename(config['Subject']['id'], config['Subject']['session'],
                           trial_num, modality=backend_modality(config))

    if not check_file_exists(trial_num):
        st.error(f"File not found: {fname}")
        return False

    hil = st.session_state.hil
    x_val = float(hil.x[hil.n, 0])
    row = hil.table.row(x_val)

    cost = st.session_state.cost_extractor.extract_cost_from_file(
        trial_num=trial_num, filename=fname)

    if cost is None or np.isnan(cost):
        reason = getattr(st.session_state.cost_extractor, 'last_failure', None)
        st.error(f"Cost extraction failed: {reason}" if reason
                 else "Cost extraction failed (no reason recorded).")
        st.caption(f"File analyzed: {fname}")
        return False

    # Non-fatal warnings used to be computed and then discarded here, so a trial
    # that silently fell back to single-sensor mode looked identical to a clean
    # one. Show them.
    for w in getattr(st.session_state.cost_extractor, 'last_warnings', []) or []:
        st.warning(w)

    if len(hil.x_opt) < 1:
        hil.x_opt = np.array([[x_val]])
        hil.y_opt = np.array([cost])
    else:
        hil.x_opt = np.concatenate((hil.x_opt, [[x_val]]))
        hil.y_opt = np.concatenate((hil.y_opt, [cost]))

    signed = _signed_mode(config)
    si_target = _si_target(config)
    phase = ("Manual ramp" if trial_num <= hil.n_ramp
             else "Bayesian Optimization")

    # "is_best" is recomputed every trial against the configured target.
    best_idx = _best_idx(hil.y_opt, config)
    this_is_best = (best_idx == len(hil.y_opt) - 1)
    for i, r in enumerate(st.session_state.results):
        r['is_best'] = (i == best_idx)

    # Store the index value AND the physical parameters it resolved to, so the
    # results CSV is readable without the table in hand — and so the record
    # survives a later regeneration of the table.
    st.session_state.results.append({
        'trial': trial_num,
        'x': x_val,
        'direction': row['direction'],
        'R': row['R'], 'theta': row['theta'],
        'L0': row['L0'], 'attach': row['attach'],
        'stiff_Nm_per_rad': row['stiff_Nm_per_rad'],
        'dose_Nm': row['dose_Nm'],
        'engage_deg': row['engage_deg'],
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
        save_dir = os.path.join(base_dir, f"sub-{subject}", f"ses-{session}",
                                backend_modality(config))
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
    if hil.n_ramp <= hil.n < n_steps:
        if config['Optimization']['normalize']:
            raw = hil.BO.run(
                hil._normalize_x(hil.x_opt).reshape(len(hil.x_opt), -1),
                hil._mean_normalize_y(hil.y_opt).reshape(len(hil.x_opt), 1))
            raw = float(hil._denormalize_x(raw).ravel()[0])
        else:
            y_for_bo = (-np.abs(hil.y_opt - si_target) if signed
                        else -hil.y_opt)
            raw = hil.BO.run(
                hil.x_opt.reshape(len(hil.x_opt), -1),
                y_for_bo.reshape(len(hil.y_opt), 1))
            raw = float(np.asarray(raw).ravel()[0])
        next_x = hil._next_x_from_table(raw)
        hil.x = np.concatenate((hil.x, [[next_x]]), axis=0)

    st.session_state.current_trial += 1
    return True


# ===========================================================================
# Plots
# ===========================================================================

def plot_index_position(x_val: float) -> go.Figure:
    """Show where this trial sits on the index, in dose and stiffness.

    Replaces the old torque-curve plot. That plot recomputed torque from a
    Python forward model that has since been deleted — it ran at a different
    spring rate with R and theta hardcoded, so it drew curves for a device the
    table does not build. The table already carries the validated numbers.

    Dose is plotted above stiffness deliberately: stiffness is the sort key,
    but dose is what the subject actually feels, and the two are not perfectly
    monotone together.
    """
    hil = st.session_state.hil
    df = hil.table.df
    row = hil.table.row(x_val)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Dose (signed ROM peak torque) — what the subject feels",
                        "Effective rotational stiffness — the sort key"),
        vertical_spacing=0.12, row_heights=[0.5, 0.5])

    fig.add_trace(go.Scatter(
        x=df['x'], y=df['dose_signed_Nm'], mode='lines+markers',
        name='dose', line=dict(color='#E8772E', width=2),
        marker=dict(size=5)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[row['x']], y=[row['dose_Nm']], mode='markers',
        name='this trial',
        marker=dict(size=16, color='#7A3A0A', symbol='diamond',
                    line=dict(color='white', width=2))), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['x'], y=df['stiff_Nm_per_rad'], mode='lines+markers',
        name='stiffness', line=dict(color='royalblue', width=2),
        marker=dict(size=5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[row['x']], y=[row['stiff_Nm_per_rad']], mode='markers',
        showlegend=False,
        marker=dict(size=16, color='#7A3A0A', symbol='diamond',
                    line=dict(color='white', width=2))), row=2, col=1)
    fig.add_hline(y=COLLINS_NM_PER_RAD, line_dash="dash", line_color="red",
                  annotation_text=f"Collins {COLLINS_NM_PER_RAD:.0f} Nm/rad",
                  row=2, col=1)

    for r in (1, 2):
        fig.add_vline(x=0, line_dash="dot", line_color="black", row=r, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=r, col=1)

    fig.update_layout(
        title=(f"Index position: x = {row['x']:+.4f}  "
               f"({_direction_label(row['direction'])})"),
        height=520, margin=dict(l=50, r=40, t=70, b=50),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=-0.18))
    fig.update_xaxes(title_text="index x   [DF ← 0 → PF]", row=2, col=1)
    fig.update_yaxes(title_text="dose (Nm)", row=1, col=1)
    fig.update_yaxes(title_text="stiffness (Nm/rad)", row=2, col=1)
    return fig


def plot_gp_surface():
    """GP posterior over the 1-D index.

    Was a pair of 3-D surfaces over (L0, attach). With one parameter this
    becomes a mean line with a confidence band — easier to read, and it shows
    which of the discrete levels the acquisition has actually spent trials on.
    """
    import torch
    hil = st.session_state.hil
    config = st.session_state.config
    if hil.n < hil.n_ramp + 1 or hil.BO.model is None:
        return None
    if len(hil.y_opt) < 2:
        return None

    x_grid = np.linspace(-1.0, 1.0, 400).reshape(-1, 1)
    x_norm = (hil._normalize_x(x_grid)
              if config['Optimization']['normalize'] else x_grid)

    hil.BO.model.eval()
    hil.BO.likelihood.eval()
    with torch.no_grad():
        pred = hil.BO.likelihood(hil.BO.model(
            torch.tensor(x_norm, dtype=torch.float64)))
        mean = pred.mean.cpu().numpy()
        std = pred.variance.sqrt().cpu().numpy()

    # The GP is fit on negated, standardized y. Undo that so the axis reads in
    # SI units rather than model-internal units.
    if config['Optimization']['normalize']:
        y_mu = np.mean(hil.y_opt)
        y_sd = np.std(hil.y_opt) if np.std(hil.y_opt) > 0 else 1.0
        mean_disp = (-mean * y_sd) + y_mu
        std_disp = std * y_sd
    else:
        mean_disp, std_disp = -mean, std

    xg = x_grid.ravel()
    best_idx = _best_idx(hil.y_opt, config)
    obs_x = hil.x_opt.ravel()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([xg, xg[::-1]]),
        y=np.concatenate([mean_disp + std_disp, (mean_disp - std_disp)[::-1]]),
        fill='toself', fillcolor='rgba(232,119,46,0.18)',
        line=dict(width=0), name='±1σ', hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=xg, y=mean_disp, mode='lines', name='GP mean',
        line=dict(color='#B5560F', width=3)))

    # Tick every reachable level, so it stays obvious that x is discrete and
    # unevenly spaced (DF steps are wider than PF steps).
    fig.add_trace(go.Scatter(
        x=hil.table.x_values,
        y=np.full(len(hil.table), float(np.min(mean_disp - std_disp))),
        mode='markers', name=f'available levels ({len(hil.table)})',
        marker=dict(symbol='line-ns-open', size=9, color='#9B7B63')))

    mask = np.ones(len(obs_x), dtype=bool)
    mask[best_idx] = False
    if mask.any():
        fig.add_trace(go.Scatter(
            x=obs_x[mask], y=hil.y_opt[mask], mode='markers', name='trials',
            marker=dict(size=11, color='#3A2415',
                        line=dict(color='white', width=1.5))))
    if _signed_mode(config):
        lbl = (f"Best (SI={hil.y_opt[best_idx]:+.2f}, "
               f"|Δ|={abs(hil.y_opt[best_idx] - _si_target(config)):.2f})")
    else:
        lbl = f"Best ({hil.y_opt[best_idx]:.2f})"
    fig.add_trace(go.Scatter(
        x=[obs_x[best_idx]], y=[hil.y_opt[best_idx]], mode='markers', name=lbl,
        marker=dict(size=17, color='#E8772E', symbol='diamond',
                    line=dict(color='#7A3A0A', width=2))))

    if _signed_mode(config):
        fig.add_hline(y=_si_target(config), line_dash="dash", line_color="red",
                      annotation_text=f"target {_si_target(config):+.1f}%")
    fig.add_vline(x=0, line_dash="dot", line_color="black",
                  annotation_text="zero torque")

    fig.update_layout(
        title="GP posterior over the index",
        xaxis_title="index x   [DF ← 0 → PF]",
        yaxis_title="SI (%)" if _signed_mode(config) else "cost",
        xaxis=dict(range=[-1.05, 1.05]),
        height=540, hovermode='x unified',
        margin=dict(l=60, r=40, t=60, b=80),
        legend=dict(orientation='h', yanchor='bottom', y=-0.28))
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
                            'Index position vs Trial'),
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

    fig.add_trace(go.Scatter(x=df['trial'], y=df['x'], mode='lines+markers',
                             name='index x', marker=dict(size=9),
                             line=dict(width=2, color='#E8772E')), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    fig.update_xaxes(title_text="Trial", row=2, col=1)
    fig.update_yaxes(title_text="SI (%)" if signed else "Cost", row=1, col=1)
    fig.update_yaxes(title_text="index x  [DF ← 0 → PF]",
                     range=[-1.05, 1.05], row=2, col=1)
    fig.update_layout(height=700, showlegend=True, hovermode='x unified')
    return fig


# ===========================================================================
# Heel-strike QC (uses hitlo.detection — always matches the BO cost)
# ===========================================================================

def analyze_trial_for_qc(xdf_path: str, cfg: DetectionConfig,
                         trim_seconds: float) -> dict:
    """Full trial analysis for QC display. Uses the SAME detection pipeline
    as the BO cost function, so what you see is what got scored."""
    # Backend-aware: the QC plot must read Trigno recordings too,
    # otherwise it silently shows nothing for a valid trial.
    left, right = load_streams(xdf_path, st.session_state.get('config'))
    if left is None or right is None:
        return None

    conf = st.session_state.get('config')
    left_result = detect_strikes(left, conf, cfg=cfg)
    right_result = detect_strikes(right, conf, cfg=cfg)

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
    hil = st.session_state.hil
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

    n_df = int(np.sum(hil.table.df['direction'] < 0))
    n_pf = int(np.sum(hil.table.df['direction'] > 0))

    st.sidebar.markdown(f"""
    **Session**
    `{config['Subject']['id']} · {config['Subject']['session']}`

    **Paradigm**
    {para}

    **Baseline** {base_line}
    **Trials** {opt['n_steps']} ({hil.n_ramp} ramp)
    **Index** x ∈ [-1, +1]
    {len(hil.table)} levels ({n_df} DF · 1 zero · {n_pf} PF)
    `{Path(hil.table.path).name}`
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
        st.success("System initialized. Use **Run Experiment** in the nav.")
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
                'index_csv': 'config/index_unified.csv',
                'n_parms': 1, 'n_steps': 15, 'manual_ramp_trials': 5,
                'ramp_sequence': [0.0, 0.2, -0.2, 0.4, -0.4],
                'range': [[-1.0], [1.0]],
                'device': 'cpu', 'normalize': True, 'acquisition': 'ei',
                'kernel_function': 'se', 'model_save_path': 'auto',
                'result_save_path': 'auto'},
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
                    "(Pre run-002), target = baseline amplified by the "
                    "displacement in whichever direction the subject already leans.")

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
            opt['manual_ramp_trials'] = st.number_input(
                "Manual ramp trials",
                value=int(opt.get('manual_ramp_trials', 5)),
                min_value=0, max_value=50, step=1,
                help="Trials run from ramp_sequence before BO takes over.")
        with oc3:
            opt['normalize'] = st.checkbox(
                "Normalize", value=bool(opt.get('normalize', True)))

        st.markdown("**Search space — the index table.** Bounds, torque caps, "
                    "and engagement filters live in `build_index_unified.m`. "
                    "To change them, regenerate the CSV; they are deliberately "
                    "not editable here, so the console and the table can never "
                    "disagree about what is safe.")

        opt['index_csv'] = st.text_input(
            "Index table CSV",
            value=str(opt.get('index_csv', 'config/index_unified.csv')))

        # These are structural, not operator choices — one parameter, fixed range.
        opt['n_parms'] = 1
        opt['range'] = [[-1.0], [1.0]]

        # Preview the table so a bad path or a stale CSV is obvious before
        # initialization rather than at trial 1.
        try:
            from hitlo.index_unified import IndexTable
            _tbl = IndexTable(opt['index_csv'])
            _df = _tbl.df
            _ndf = int(np.sum(_df['direction'] < 0))
            _npf = int(np.sum(_df['direction'] > 0))
            _pf_rad = _df.loc[_df['direction'] > 0, 'stiff_Nm_per_rad']
            _df_rad = _df.loc[_df['direction'] < 0, 'stiff_Nm_per_rad'].abs()
            st.success(
                f"Loaded {len(_tbl)} levels — {_ndf} DF · 1 zero · {_npf} PF.  "
                f"PF {_pf_rad.min():.0f}–{_pf_rad.max():.0f} Nm/rad,  "
                f"DF {_df_rad.min():.0f}–{_df_rad.max():.0f} Nm/rad.")
            if _pf_rad.max() >= COLLINS_NM_PER_RAD >= _pf_rad.min():
                st.caption(f"PF arm brackets the Collins "
                           f"{COLLINS_NM_PER_RAD:.0f} Nm/rad optimum.")
            else:
                st.warning(f"PF arm does NOT bracket Collins "
                           f"{COLLINS_NM_PER_RAD:.0f} Nm/rad.")
            _table_ok = True
        except Exception as e:
            st.error(f"Could not load index table: {e}")
            _tbl = None
            _table_ok = False

        ramp_str = st.text_input(
            "Ramp sequence (x values, comma-separated)",
            value=", ".join(str(v) for v in
                            opt.get('ramp_sequence', [0.0, 0.2, -0.2, 0.4, -0.4])),
            help="Snapped to real table rows at load. Touch BOTH arms — if "
                 "every ramp trial sits on one side, the GP starts blind on "
                 "the other.")
        try:
            opt['ramp_sequence'] = [float(v.strip())
                                    for v in ramp_str.split(',') if v.strip()]
        except ValueError:
            st.error("Ramp sequence must be comma-separated numbers.")

        _ramp = opt.get('ramp_sequence', [])
        if len(_ramp) != int(opt.get('manual_ramp_trials', 0)):
            st.warning(
                f"Ramp sequence has {len(_ramp)} values but "
                f"manual_ramp_trials = {opt.get('manual_ramp_trials')}.")
        if _ramp and _table_ok:
            _snapped = [_tbl.snap(v) for v in _ramp]
            _n_df_ramp = sum(1 for v in _snapped if v < 0)
            _n_pf_ramp = sum(1 for v in _snapped if v > 0)
            st.caption("Snaps to: " +
                       ",  ".join(f"{v:+.4f}" for v in _snapped))
            if _n_df_ramp == 0 or _n_pf_ramp == 0:
                st.warning(
                    f"Ramp covers only one arm ({_n_df_ramp} DF, "
                    f"{_n_pf_ramp} PF). The GP will have no data on the other "
                    f"side when BO starts.")

    st.markdown("---")
    bc1, bc2, bc3 = st.columns([1, 1, 1])
    with bc1:
        if st.button("💾 Save config", type="primary", width="stretch"):
            if save_config_to_disk(_cfg):
                st.session_state.config_saved = True
                st.success("Saved.")
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
                st.success("Initialized. Go to **Run Experiment**.")
                st.rerun()

    if not st.session_state.config_saved:
        st.caption("Save (or use existing) config to enable Initialize.")

    with st.expander("📄 Preview YAML"):
        st.code(yaml.safe_dump(_cfg, sort_keys=False), language="yaml")

    if _table_ok:
        with st.expander("📋 Index table"):
            st.dataframe(_tbl.df, width="stretch", height=400)


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
    section_chip("Live monitor")
    st.subheader("📡 Live Polar H10 — Left + Right Shank")
    st.caption("Monitoring only. Scanning, assignment and the leg check live on "
               "the **Sensors** page.")
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
                    update_live_data(inlet, store, side)
                    st.success(f"{side.capitalize()} · "
                               f"{len(store['time'])} samples")
                    fig = plot_live_sensor(store, f'polar accel {side}', sr)
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info(f"Collecting {side}…")
                else:
                    st.warning(f"{side.capitalize()} not found")
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
        f"Target = baseline amplified by the displacement, in its own direction.")

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
            st.success(f"Found {base_fn}")
        else:
            st.warning(f"Waiting for {base_fn}")
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
        base = float(st.session_state.baseline_si)
        pv = compute_baseline_target(base, float(disp))
        if abs(base) < BASELINE_DEADBAND_PCT:
            st.caption(f"→ baseline {base:+.2f}% is within ±{BASELINE_DEADBAND_PCT}% "
                       f"of zero (no resolvable direction) — using the device's "
                       f"default direction. Target = **{pv:+.2f}%**")
        else:
            arrow = "more positive" if base > 0 else "more negative"
            st.caption(f"→ baseline {base:+.2f}%, amplified {disp:.0f}% "
                       f"{arrow} = **{pv:+.2f}%**")


def _optimization_phase(config, signed):
    hil = st.session_state.hil
    trial_num = st.session_state.current_trial + 1
    n_steps = config['Optimization']['n_steps']
    n_ramp = hil.n_ramp
    si_target = _si_target(config)

    if trial_num > n_steps:
        st.success("🎉 OPTIMIZATION COMPLETE — see **Results**.")
        return

    section_chip(f"Step 3 · Trial {trial_num} of {n_steps}")
    c1, c2 = st.columns([2, 1])
    with c1:
        if trial_num <= n_ramp:
            st.subheader(f"📐 Manual ramp {trial_num}/{n_ramp}")
        else:
            st.subheader(f"🧠 Bayesian Optimization "
                         f"{trial_num - n_ramp}/{n_steps - n_ramp}")
    with c2:
        st.progress((trial_num - 1) / n_steps,
                    text=f"{int((trial_num-1)/n_steps*100)}%")

    # Generate next BO suggestion if needed
    if hil.n >= len(hil.x):
        if len(hil.x_opt) >= n_ramp:
            try:
                if config['Optimization']['normalize']:
                    raw = hil.BO.run(
                        hil._normalize_x(hil.x_opt).reshape(len(hil.x_opt), -1),
                        hil._mean_normalize_y(hil.y_opt).reshape(len(hil.x_opt), 1))
                    raw = float(hil._denormalize_x(raw).ravel()[0])
                else:
                    yb = (-np.abs(hil.y_opt - si_target) if signed
                          else -hil.y_opt)
                    raw = hil.BO.run(hil.x_opt.reshape(len(hil.x_opt), -1),
                                     yb.reshape(len(hil.y_opt), -1))
                    raw = float(np.asarray(raw).ravel()[0])
                hil.x = np.concatenate(
                    (hil.x, [[hil._next_x_from_table(raw)]]), axis=0)
            except Exception as e:
                st.error(f"BO suggestion failed: {e}")
                return
        else:
            st.error("Parameter index out of bounds — reinitialize.")
            return

    x_val = float(hil.x[hil.n, 0])
    row = hil.table.row(x_val)
    direction = _direction_label(row['direction'])

    st.markdown(f"### Set on device — index x = {x_val:+.4f}  ({direction})")

    if row['is_zero']:
        st.warning(
            "**ZERO TORQUE CONDITION.** This is NOT device-off: the 24 bands "
            "are always on, so the controller must actively cancel them "
            "(τ_cable = 0 − τ_band(θ)).")
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("R (m)", f"{row['R']:.4f}")
        p2.metric("θ (deg)", f"{row['theta']:.2f}")
        p3.metric("L₀ (m)", f"{row['L0']:.4f}")
        p4.metric("attach", f"{row['attach']:+.4f}")
        d1, d2, d3 = st.columns(3)
        d1.metric("Stiffness", f"{row['stiff_Nm_per_rad']:+.1f} Nm/rad")
        d2.metric("Dose in ROM", f"{row['dose_Nm']:+.2f} Nm")
        d3.metric("Engages at", f"{row['engage_deg']:+.1f}°")

    st.plotly_chart(plot_index_position(x_val), width="stretch")

    with st.expander("📝 LabRecorder steps", expanded=True):
        if row['is_zero']:
            setup_line = ("Set the controller to cancel the bands "
                          "(τ_cable = 0 − τ_band(θ)). Do not simply remove them.")
        else:
            setup_line = (
                f"Set **R = {row['R']:.4f} m**, **θ = {row['theta']:.2f}°**, "
                f"**L₀ = {row['L0']:.4f} m**, **attach = {row['attach']:+.4f}** "
                f"on Computer 2.")
            if row['direction'] < 0:
                setup_line += (" DF condition — the controller renders this as "
                               "τ_cable = τ_desired − τ_band(θ), since the "
                               "bands are always on.")
        st.markdown(f"""
        1. {setup_line}
        2. LabRecorder: Block/Task = `Default`, Run = `{trial_num}` →
           `sub-{config['Subject']['id']}_ses-{config['Subject']['session']}_task-Default_run-{trial_num:03d}_eeg.xdf`
        3. Walk {config['Cost']['time']} s · Stop · Analyze below.
        """)

    a1, a2, a3 = st.columns(3)
    with a1:
        ok = check_file_exists(trial_num)
        if ok:
            st.success(f"run-{trial_num:03d}.xdf")
        else:
            st.warning(f"Waiting for run-{trial_num:03d}.xdf")
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
                            config['Subject']['session'], last,
                            modality=backend_modality(config))
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
                    st.success("QC clean.")
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
        if hil.n < hil.n_ramp + 1 or hil.BO.model is None:
            st.info(f"Live GP appears after the ramp ({hil.n_ramp} trials) "
                    f"+ 1 BO trial. Currently {hil.n} trials done.")
            return
        fig = plot_gp_surface()
        if fig:
            st.plotly_chart(fig, width="stretch")
        if st.session_state.results:
            st.markdown("---")
            section_chip("Trial history")
            df = pd.DataFrame(st.session_state.results)
            st.dataframe(df[['trial', 'x', 'dose_Nm', 'stiff_Nm_per_rad',
                             'cost', 'phase']], width="stretch")
        return

    # ---- Load past session ----
    base_default = (st.session_state.config['Subject']['base_dir']
                    if st.session_state.get('config')
                    else '/Users/maccamardo/HITLO_Data')
    c1, c2, c3 = st.columns(3)
    with c1:
        base_dir = st.text_input("Base dir", value=base_default)
    with c2:
        subject = st.text_input("Subject", value="P062")
    with c3:
        session = st.text_input("Session", value="S001")

    _gp_historical_viewer(base_dir, subject, session)


def _gp_historical_viewer(base_dir, subject, session):
    """Load saved GP checkpoints from disk and scrub iterations.

    Handles BOTH parameterizations. Sessions run before August 2026 saved a 2-D
    GP over (L0, attach); sessions from the unified index on save a 1-D GP over
    x. The saved data.csv column count tells them apart — there is no flag in
    the file, and old sessions cannot be re-rendered as 1-D, so both renderers
    have to stay. Do not "simplify" this by deleting the 2-D branch: that is the
    only way back into P048/P062-era models.
    """
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

    st.success(f"{len(hil_results)} trials · {len(iters)} checkpoints "
               f"(iter {min(iters)}–{max(iters)})")
    sel = st.select_slider("Iteration", options=iters, value=iters[0])

    ipath = models_dir / f'iter_{sel}'
    try:
        data = np.loadtxt(ipath / 'data.csv')
        if data.ndim == 1:
            data = data.reshape(1, -1)
    except Exception as e:
        st.error(f"Could not load iteration {sel}: {e}")
        return

    n_parms = data.shape[1] - 1  # last column is the BO objective

    if n_parms == 1:
        st.caption("1-D session (unified index).")
        _hist_1d(data, hil_results, ipath, sel, torch, SingleTaskGP,
                 GaussianLikelihood)
    elif n_parms == 2:
        st.caption("2-D session — legacy (L₀, attach) parameterization. "
                   "Predates the unified index.")
        _hist_2d(data, hil_results, ipath, sel, torch, SingleTaskGP,
                 GaussianLikelihood)
    else:
        st.error(f"Unrecognized checkpoint shape: {data.shape[1]} columns.")


def _hist_1d(data, hil_results, ipath, sel, torch, SingleTaskGP,
             GaussianLikelihood):
    """Render a saved 1-D GP over the unified index."""
    xn, bo = data[:, 0], data[:, 1]
    xp = xn * 2.0 - 1.0                      # [0,1] -> [-1,+1]
    actual = hil_results['cost'].values[:sel]

    grid_n = np.linspace(0.0, 1.0, 400).reshape(-1, 1)
    grid_p = grid_n.ravel() * 2.0 - 1.0
    try:
        Xtr = torch.tensor(xn.reshape(-1, 1), dtype=torch.float64)
        ytr = torch.tensor(bo, dtype=torch.float64).reshape(-1, 1)
        lik = GaussianLikelihood()
        model = SingleTaskGP(Xtr, ytr, likelihood=lik)
        model.load_state_dict(torch.load(ipath / 'model.pth'), strict=False)
        model.eval()
        with torch.no_grad():
            pred = lik(model(torch.tensor(grid_n, dtype=torch.float64)))
            mean = pred.mean.numpy()
            lo, up = pred.confidence_region()
            lo, up = lo.numpy(), up.numpy()
    except Exception as e:
        st.error(f"Could not rebuild GP for iteration {sel}: {e}")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Iteration", sel)
    m2.metric("Trials so far", len(xp))
    m3.metric("|Best SI|", f"{np.abs(actual).min():.2f}%"
              if len(actual) else "—")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([grid_p, grid_p[::-1]]),
        y=np.concatenate([up, lo[::-1]]),
        fill='toself', fillcolor='rgba(232,119,46,0.18)',
        line=dict(width=0), name='confidence', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=grid_p, y=mean, mode='lines', name='GP mean',
                             line=dict(color='#B5560F', width=3)))
    fig.add_trace(go.Scatter(
        x=xp, y=bo, mode='markers+text',
        marker=dict(size=10, color='#3A2415',
                    line=dict(color='white', width=1.5)),
        text=[str(i + 1) for i in range(len(xp))],
        textposition='top center', textfont=dict(size=9), name='trials'))
    fig.add_vline(x=0, line_dash='dot', line_color='black')
    fig.update_layout(
        height=520, hovermode='x unified',
        xaxis_title='index x   [DF ← 0 → PF]',
        yaxis_title='BO objective (model units)',
        xaxis=dict(range=[-1.05, 1.05]),
        margin=dict(l=60, r=40, t=40, b=60))
    st.plotly_chart(fig, width="stretch")

    with st.expander("📊 Trial data"):
        n = min(len(xp), len(bo), len(actual))
        if n == 0:
            st.info("No overlapping trial data for this iteration.")
        else:
            st.dataframe(pd.DataFrame({
                'Trial': range(1, n + 1),
                'x': xp[:n],
                'BO cost': bo[:n],
                'signed SI %': actual[:n]}), width="stretch")


def _hist_2d(data, hil_results, ipath, sel, torch, SingleTaskGP,
             GaussianLikelihood):
    """Render a saved 2-D GP from the legacy (L₀, attach) parameterization.

    The ranges come from the results CSV rather than the live config, because
    the live config now describes a 1-D space and would denormalize these
    coordinates into nonsense.
    """
    L0n, attachn, bo = data[:, 0], data[:, 1], data[:, 2]
    actual = hil_results['cost'].values[:sel]

    if {'L0', 'attach'}.issubset(hil_results.columns):
        L0_MIN = float(hil_results['L0'].min())
        L0_MAX = float(hil_results['L0'].max())
        attach_MIN = float(hil_results['attach'].min())
        attach_MAX = float(hil_results['attach'].max())
        if L0_MAX - L0_MIN < 1e-9 or attach_MAX - attach_MIN < 1e-9:
            L0_MIN, L0_MAX = 0.30, 0.38
            attach_MIN, attach_MAX = -0.25, 0.75
            st.caption("Degenerate observed range — using v2.5.1 defaults.")
        else:
            st.caption("Axis ranges inferred from this session's results CSV.")
    else:
        L0_MIN, L0_MAX = 0.30, 0.38
        attach_MIN, attach_MAX = -0.25, 0.75
        st.caption("No L₀/attach columns in results — using v2.5.1 defaults.")

    L0p = L0n * (L0_MAX - L0_MIN) + L0_MIN
    attachp = attachn * (attach_MAX - attach_MIN) + attach_MIN

    L0g = np.linspace(L0_MIN, L0_MAX, 40)
    attachg = np.linspace(attach_MIN, attach_MAX, 40)
    LL, AA = np.meshgrid(L0g, attachg)
    Xt = np.column_stack([((LL - L0_MIN) / (L0_MAX - L0_MIN)).ravel(),
                          ((AA - attach_MIN) / (attach_MAX - attach_MIN)).ravel()])
    try:
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
        st.error(f"Could not rebuild GP for iteration {sel}: {e}")
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
        text=[str(i + 1) for i in range(_ns)],
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
    best = st.session_state.results[best_idx]
    best_cost = best['cost']
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
        st.metric("Latest SI" if signed else "Latest",
                  f"{latest:+.2f}%" if signed else f"{latest:.3f}")
    with m4:
        if st.session_state.baseline_si is not None:
            st.metric("Baseline", f"{st.session_state.baseline_si:+.2f}%")
        else:
            st.metric("Baseline", "n/a")

    # Which configuration actually won, in physical terms.
    section_chip("Best configuration")
    if best.get('direction', 0) == 0:
        st.info(f"**x = {best['x']:+.4f} — ZERO TORQUE** was the best trial. "
                f"The subject's symmetry was closest to target with the device "
                f"cancelled.")
    else:
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("index x", f"{best['x']:+.4f}")
        b2.metric("R (m)", f"{best['R']:.4f}")
        b3.metric("θ (deg)", f"{best['theta']:.2f}")
        b4.metric("L₀ (m)", f"{best['L0']:.4f}")
        b5.metric("attach", f"{best['attach']:+.4f}")
        st.caption(f"{_direction_label(best['direction'])} · "
                   f"{best['stiff_Nm_per_rad']:+.1f} Nm/rad · "
                   f"dose {best['dose_Nm']:+.2f} Nm in ROM")

    # ------------------------------------------------------------------
    # What to physically set for the post-optimization training phases.
    # This is the number the operator actually needs: after the 15 trials the
    # protocol runs 10 min assistive + 10 min EA at "the best result", and
    # "best" has to mean something defensible.
    # ------------------------------------------------------------------
    section_chip("Carry into the training phase")
    hil = st.session_state.get('hil')
    post = hil.posterior_best() if hil is not None else None

    if post is None:
        st.info("GP not fitted yet — best observed trial is the only estimate.")
    else:
        agree = abs(post['x'] - post['best_observed_x']) < 1e-9
        pr = post['row']

        # ---- the three validity checks, computed before anything is claimed --
        spread = float(post['mu'].max() - post['mu'].min())
        unc = float(np.mean(post['sd']))
        t1_ratio = spread / max(unc, 1e-9)
        t1 = t1_ratio > 2.0

        rows_df = pd.DataFrame(st.session_state.results)
        REPEAT_TOL = 0.10      # configs within 10% stiffness count as repeats
        REPEAT_MAX = 5.0       # pts; above this, repeats do not reproduce
        worst_repeat, n_groups = 0.0, 0
        if 'stiff_Nm_per_rad' in rows_df and len(rows_df) > 2:
            st_v = rows_df['stiff_Nm_per_rad'].to_numpy(float)
            for i in range(len(st_v)):
                near = np.abs(st_v - st_v[i]) <= REPEAT_TOL * max(abs(st_v[i]), 1.0)
                if near.sum() >= 2:
                    n_groups += 1
                    sp = float(rows_df.loc[near, 'cost'].max()
                               - rows_df.loc[near, 'cost'].min())
                    worst_repeat = max(worst_repeat, sp)
        t2 = n_groups > 0 and worst_repeat <= REPEAT_MAX

        zero_rows = rows_df[rows_df.get('direction', pd.Series(dtype=int)) == 0]
        t3 = None
        gain = zero_gap = None
        if len(zero_rows):
            zero_gap = float(abs(zero_rows['cost'].iloc[0] - si_target))
            gain = zero_gap - post['best_observed_distance']
            t3 = gain > max(worst_repeat, REPEAT_MAX)

        resolved = bool(t1 and t2 and (t3 is not False))

        # ---- WHAT TO SET — first, unmissable ------------------------------
        if resolved:
            st.success("### ✅ Set the device to this for the 10-min phases")
        else:
            st.warning("### ⚠️ Set the device to this for the 10-min phases")
            st.caption("This is the best available **estimate**, not a located "
                       "optimum — see the checks below. You still carry it "
                       "forward; you just do not report it as the optimum.")
        if pr['is_zero']:
            st.warning("ZERO TORQUE is the estimated optimum — the controller "
                       "must actively cancel the bands, this is not device-off.")
        else:
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("index x", f"{post['x']:+.4f}")
            k2.metric("R (m)", f"{pr['R']:.4f}")
            k3.metric("θ (deg)", f"{pr['theta']:.2f}")
            k4.metric("L₀ (m)", f"{pr['L0']:.4f}")
            k5.metric("attach", f"{pr['attach']:+.4f}")
            st.caption(f"{_direction_label(pr['direction'])} · "
                       f"{pr['stiff_Nm_per_rad']:+.1f} Nm/rad · "
                       f"dose {pr['dose_Nm']:+.2f} Nm"
                       + ("" if post['was_tested'] else
                          " · this level was NEVER TESTED — the GP inferred it "
                          "from neighbouring trials"))

        c1, c2 = st.columns(2)
        c1.metric("GP posterior argmin",
                  f"{post['x']:+.4f}",
                  help="The model's belief, pooling every trial. Standard "
                       "estimator in HILBO work.")
        c2.metric("Best observed trial",
                  f"{post['best_observed_x']:+.4f}",
                  f"SI {post['best_observed_si']:+.2f}%",
                  help="argmin over trials actually walked. Biased low: taking "
                       "the minimum of noisy draws selects the most favourable "
                       "error, not the best underlying value.")

        if agree:
            st.success("Both estimators agree — carry this configuration forward.")
        else:
            st.warning(
                f"**They disagree** (Δx = {abs(post['x']-post['best_observed_x']):.3f}). "
                f"The GP does not believe the luckiest draw. Prefer the posterior "
                f"argmin, and report both — a gap this size is itself a statement "
                f"about your noise level.")

        st.caption(f"predicted |SI − target| at this level: "
                   f"**{post['predicted_distance']:.2f} ± {post['predicted_sd']:.2f}** pts")

        # ---- is this actually an optimum? ---------------------------------
        st.markdown("---")
        st.markdown("**Is this a located optimum, or the best guess?**")

        def _row(ok, name, detail):
            icon = "✅" if ok else ("❌" if ok is False else "➖")
            st.markdown(f"{icon} **{name}** — {detail}")

        _row(t1, "Posterior is resolved",
             f"spread {spread:.3f} vs mean uncertainty {unc:.3f} "
             f"(ratio {t1_ratio:.2f}, need > 2). "
             + ("The GP distinguishes levels."
                if t1 else "The curve is flat inside its own error — the model "
                           "cannot tell levels apart."))
        _row(t2, "Repeats reproduce",
             (f"worst spread between configs within {REPEAT_TOL*100:.0f}% "
              f"stiffness: {worst_repeat:.1f} pts (need < {REPEAT_MAX:.0f}). "
              + ("Setting the device the same way gives the same answer."
                 if t2 else "Setting the device the same way gives different "
                            "answers, by more than the effect you are chasing."))
             if n_groups else "no repeated configurations in this session")
        if t3 is None:
            _row(None, "Beats the no-torque control",
                 "no zero-torque trial in this session")
        else:
            _row(t3, "Beats the no-torque control",
                 f"improves on zero torque by {gain:.1f} pts "
                 f"(zero missed by {zero_gap:.1f}, this level by "
                 f"{post['best_observed_distance']:.1f}); needs to exceed the "
                 f"{max(worst_repeat, REPEAT_MAX):.1f} pt repeat spread.")

        if resolved:
            st.success("**All checks pass** — reportable as a subject-specific optimum.")
        else:
            claim = ("You can report that the optimized condition improved "
                     "symmetry tracking relative to the null condition."
                     if t3 else "")
            st.error(
                "**Not a located optimum.** Carry the configuration above into "
                "the training phases — you have to set something, and this is the "
                "best estimate — but do not write that a subject-specific optimum "
                "was identified. " + claim)
            if n_groups and not t2:
                st.caption("The repeat check is the gate, and it is a measurement "
                           "problem rather than an algorithm one: longer trials, a "
                           "settling window, and deliberate repeat trials shrink "
                           "that spread. Under ~5 pts the posterior check usually "
                           "passes on its own.")

        with st.expander("Why not just use the last BO suggestion?"):
            st.markdown(
                "The acquisition function chooses where to **sample next**, "
                "balancing exploration against exploitation — it is not a claim "
                "about where the optimum is. Late in a run it is often still "
                "exploring, or (as in sub-P997) its expected improvement has "
                "collapsed to ~0 and the remaining trials are effectively "
                "arbitrary. The final query point carries no special status.\n\n"
                "**Caveat for this device:** a single GP spanning the ~15x "
                "PF/DF stiffness asymmetry under-resolves the dorsiflexor arm. "
                "If the optimum sits on the DF side, the posterior mean is least "
                "trustworthy exactly where it matters, and two per-arm GPs would "
                "be the fix.")

    # Which arm the trials went to. Worth watching across subjects: if DF keeps
    # winning, the single GP is under-resolving the arm that matters and the
    # two-GP split becomes worth building.
    xs = np.array([r['x'] for r in st.session_state.results])
    n_df = int(np.sum(xs < 0))
    n_pf = int(np.sum(xs > 0))
    n_zero = int(np.sum(np.isclose(xs, 0.0)))
    st.caption(f"Trials by arm: **{n_df} DF** · {n_zero} zero · **{n_pf} PF**")

    if (fig := plot_progress()):
        st.plotly_chart(fig, width="stretch")

    if len(st.session_state.results) > hil.n_ramp:
        section_chip("GP posterior")
        if (gp := plot_gp_surface()):
            st.plotly_chart(gp, width="stretch")

    section_chip("Trial history")
    df = pd.DataFrame(st.session_state.results)
    disp = df.copy()
    disp['x'] = disp['x'].map('{:+.4f}'.format)
    disp['dose_Nm'] = disp['dose_Nm'].map('{:+.2f}'.format)
    disp['stiff_Nm_per_rad'] = disp['stiff_Nm_per_rad'].map('{:+.1f}'.format)
    disp['cost'] = disp['cost'].map(
        '{:+.4f}'.format if signed else '{:.4f}'.format)
    if 'dist_from_target' in disp.columns:
        disp['|Δ target|'] = disp['dist_from_target'].map('{:.4f}'.format)
    disp['best'] = disp['is_best'].map(lambda x: '⭐' if x else '')
    cols = ['trial', 'x', 'stiff_Nm_per_rad', 'dose_Nm', 'cost']
    if '|Δ target|' in disp.columns and signed:
        cols.append('|Δ target|')
    cols += ['phase', 'best']
    st.dataframe(disp[cols], width="stretch", height=360)

    with st.expander("🔧 Physical parameters per trial"):
        phys = df[['trial', 'x', 'R', 'theta', 'L0', 'attach',
                   'engage_deg']].copy()
        st.dataframe(phys, width="stretch")

    csv = df.to_csv(index=False)
    st.download_button("💾 Download results CSV", data=csv,
                       file_name=f"{config['Subject']['id']}_results.csv",
                       mime="text/csv", type="primary")



# ===========================================================================
# SENSORS PAGE — scan, assign, stream, and SEE which leg is which
# ===========================================================================

SENSORS_JSON = REPO_ROOT / 'config' / 'sensors.json'


def _load_sensor_ids() -> dict:
    if SENSORS_JSON.is_file():
        try:
            d = json.loads(SENSORS_JSON.read_text())
            return {'left': str(d['left']).upper(), 'right': str(d['right']).upper()}
        except Exception:
            pass
    return {'left': '', 'right': ''}


def _save_sensor_ids(ids: dict) -> None:
    SENSORS_JSON.parent.mkdir(parents=True, exist_ok=True)
    SENSORS_JSON.write_text(json.dumps(ids, indent=2) + "\n")


def _scan_polar(timeout: float = 12.0) -> list:
    """Shell out to collect_sensors --scan and parse the table it prints."""
    import subprocess, re
    r = subprocess.run([sys.executable, str(REPO_ROOT/'apps'/'collect_sensors.py'),
                        '--scan', '--scan-timeout', str(timeout)],
                       capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        m = re.match(r'\s*\d+\s+([0-9A-F]{8})\s+(-?\d+|\s*n/a)\s*dBm?\s+(.*)', line)
        if m:
            rssi = m.group(2).strip()
            out.append({'id': m.group(1),
                        'rssi': int(rssi) if rssi.lstrip('-').isdigit() else None,
                        'name': m.group(3).strip()})
    return out


def _streamer_running(side: str) -> bool:
    p = st.session_state.get(f'proc_{side}')
    return p is not None and p.poll() is None


def _start_streamer(side: str):
    import subprocess
    if _streamer_running(side):
        return
    env = {**os.environ, 'PYTHONUNBUFFERED': '1',
           'LSLAPICFG': str(REPO_ROOT/'config'/'lsl_api.cfg')}
    st.session_state[f'proc_{side}'] = subprocess.Popen(
        [sys.executable, '-u', str(REPO_ROOT/'apps'/'collect_sensors.py'),
         side, '--scan-timeout', '25'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=env, cwd=str(REPO_ROOT))


def _stop_streamer(side: str):
    p = st.session_state.get(f'proc_{side}')
    if p is not None and p.poll() is None:
        p.terminate()
    st.session_state[f'proc_{side}'] = None


def _motion_both(seconds: float = 0.6):
    """Motion on both sides over the same window: {'left': v, 'right': v}.

    Both backends are handled here rather than per side, because on Trigno the
    two sides share ONE inlet — pulling it once per side would race, with each
    pull consuming samples the other never sees.
    """
    import time as _t
    li = st.session_state.get('lsl_inlet_left')
    ri = st.session_state.get('lsl_inlet_right')
    if li is None or ri is None:
        return {}
    cols = st.session_state.get('trigno_cols')
    shared = cols is not None and li is ri
    mags = {'left': [], 'right': []}
    t0 = _t.time()
    while _t.time() - t0 < seconds:
        if shared:
            chunk, _ = li.pull_chunk(timeout=0.05, max_samples=256)
            for smp in chunk:
                for sd in ('left', 'right'):
                    cx, cy, cz = cols[sd]
                    if len(smp) > max(cx, cy, cz):
                        mags[sd].append(
                            (smp[cx]**2 + smp[cy]**2 + smp[cz]**2) ** 0.5)
        else:
            for sd, inl in (('left', li), ('right', ri)):
                chunk, _ = inl.pull_chunk(timeout=0.05, max_samples=256)
                for smp in chunk:
                    if len(smp) >= 3:
                        mags[sd].append(
                            (smp[0]**2 + smp[1]**2 + smp[2]**2) ** 0.5)
    return {sd: (float(np.std(np.asarray(v))) if len(v) >= 5 else None)
            for sd, v in mags.items()}


def page_sensors():
    """Sensor setup. The two backends need entirely different pages.

    Polar sensors are paired over BLE from this machine, so the page scans,
    assigns sides and starts per-sensor processes. Trigno sensors are paired
    in the Trigno Control Utility on the Windows machine and reach us as one
    already-assembled LSL stream, so there is nothing here to scan or start —
    only to verify.
    """
    if _backend() == 'trigno':
        return _page_sensors_trigno()
    return _page_sensors_polar()


def _page_sensors_trigno():
    section_chip("Step 0 · Sensors")
    st.header("🛰️ Sensors — Trigno")
    cfg = _active_config()
    want = (cfg.get('Sensing') or {}).get('stream', 'TrignoIMU')
    st.caption(f"Pairing and slot assignment happen in the Trigno Control "
               f"Utility on the base-station machine. This page verifies what "
               f"arrives as **{want}**.")

    # ---- 1. is the stream there, and what is in it ----
    st.subheader("1 · Stream")
    if st.button("🔍 Look for the stream (3 s)", width="stretch"):
        st.session_state.trigno_probe = _probe_trigno_stream(want)
    probe = st.session_state.get('trigno_probe')
    if probe is None:
        st.info(f"Not checked yet. The bridge must be running on the "
                f"base-station machine and publishing **{want}**.")
    elif probe.get('error'):
        st.error(probe['error'])
    else:
        a, b, c = st.columns(3)
        a.metric("channels", probe['n_channels'])
        b.metric("rate", f"{probe['srate']:.0f} Hz")
        c.metric("host", probe['host'])
        inv = probe['inventory']
        if inv:
            st.markdown("**Sensors detected, by label:**")
            for side in ('left', 'right'):
                segs = inv.get(side, [])
                if segs:
                    st.markdown(f"- **{side}** — " + ", ".join(sorted(segs)))
                else:
                    st.warning(f"- **{side}** — nothing found")
            missing = [sd for sd in ('left', 'right')
                       if 'shank' not in inv.get(sd, [])]
            if missing:
                st.error(
                    f"No shank sensor for: {', '.join(missing)}. Step-time "
                    f"symmetry needs a shank sensor on BOTH legs — extra "
                    f"segments (foot, thigh) are carried but not used by the "
                    f"cost function.")
            else:
                st.success("Both shanks present — the cost function has what "
                           "it needs.")
        else:
            st.error(
                "The stream declares no usable channel labels. Left and right "
                "cannot be separated without them, and guessing a column order "
                "would swap the legs — which inverts the symmetry index while "
                "producing entirely plausible numbers. Fix the labels in the "
                "bridge (left_shank_gyr_z, right_foot_acc_x, ...).")

    st.markdown("---")

    # ---- 2. live attachment ----
    st.subheader("2 · Live data")
    connect_to_lsl()
    err = st.session_state.get('trigno_label_error')
    if err:
        st.error(err)
    live = (st.session_state.get('lsl_inlet_left') is not None and
            st.session_state.get('lsl_inlet_right') is not None)
    if live:
        st.success("Attached — both sides are being read from the stream.")
    else:
        st.warning("Not attached. Start the bridge, then reload this page.")
    if st.button("🔄 Re-attach"):
        st.session_state.lsl_inlet_left = None
        st.session_state.lsl_inlet_right = None
        st.session_state.trigno_cols = None
        connect_to_lsl()
        st.rerun()

    st.markdown("---")

    # ---- 3. the leg check ----
    st.subheader("3 · Confirm which leg is which")
    st.caption("Shake one leg at a time and watch which bar responds. On "
               "Trigno the side comes from the **bridge's slot-to-label "
               "mapping**, not from anything on this machine — so if the wrong "
               "bar moves, the fix is in the bridge, not here.")
    _live_bars_shared()

    st.markdown("---")
    st.subheader("4 · Preflight")
    if st.button("🚦 Run preflight"):
        import subprocess
        r = subprocess.run([sys.executable, str(REPO_ROOT / 'apps' / 'preflight.py')],
                           capture_output=True, text=True, cwd=str(REPO_ROOT))
        (st.success if r.returncode == 0 else st.error)(
            "READY" if r.returncode == 0 else "NOT READY")
        st.code(r.stdout or r.stderr, language=None)


def _probe_trigno_stream(want: str) -> dict:
    """One look at the named LSL stream: shape, rate, host, and inventory."""
    try:
        from pylsl import resolve_streams
        for s in resolve_streams(wait_time=3.0):
            if s.name() != want:
                continue
            labels = []
            try:
                ch = s.desc().child('channels').child('channel')
                while not ch.empty():
                    labels.append(ch.child_value('label').strip().lower())
                    ch = ch.next_sibling()
            except Exception:
                pass
            inv = {}
            for lab in labels:
                for side in ('left', 'right'):
                    if lab.startswith(side):
                        seg = ('foot' if 'foot' in lab else
                               'thigh' if 'thigh' in lab else 'shank')
                        inv.setdefault(side, set()).add(seg)
            return {'n_channels': s.channel_count(), 'srate': s.nominal_srate(),
                    'host': s.hostname(), 'inventory': {k: sorted(v) for k, v in inv.items()},
                    'error': None}
        return {'error': f"No stream named '{want}' on the network. Is the "
                         f"bridge running on the base-station machine?"}
    except Exception as e:
        return {'error': f"{type(e).__name__}: {e}"}


def _live_bars_shared():
    @st.fragment(run_every=0.8)
    def _bars():
        connect_to_lsl()
        if (st.session_state.get('lsl_inlet_left') is None or
                st.session_state.get('lsl_inlet_right') is None):
            st.info("Waiting for the stream — the bars appear as soon as data "
                    "is flowing.")
            return
        mv = _motion_both()
        ml = mv.get('left') or 0.0
        mr = mv.get('right') or 0.0
        top = max(ml, mr, 1.0)
        f1, f2 = st.columns(2)
        for col, name, val in ((f1, 'LEFT', ml), (f2, 'RIGHT', mr)):
            with col:
                moving = val > 2.5 * min(ml, mr) and val > 60
                st.markdown(f"**{name}** {'🟢 MOVING' if moving else ''}")
                st.progress(min(val / top, 1.0))
                st.caption(f"motion {val:.0f}")
    _bars()


def _page_sensors_polar():
    section_chip("Step 0 · Sensors")
    st.header("🛰️ Sensors — Polar H10")
    st.caption("Get both straps streaming, then confirm which LEG each one is on. "
               "A device id names a strap, not the limb it ended up on.")

    ids = _load_sensor_ids()

    # ---- scan + assign ----
    st.subheader("1 · Which straps are in range")
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("🔍 Scan (12 s)", width="stretch"):
            with st.spinner("scanning for Polar H10 ..."):
                st.session_state.scan_results = _scan_polar()
    found = st.session_state.get('scan_results', [])
    with c2:
        if found:
            df = pd.DataFrame(found)[['id', 'rssi', 'name']]
            df.columns = ['Device ID', 'RSSI (dBm)', 'Name']
            st.dataframe(df, hide_index=True, width="stretch")
        else:
            st.info("No scan yet. Straps must be **wet and worn** — the H10 sleeps "
                    "until its electrodes are bridged by skin.")

    # The lab has more straps than this experiment uses and they get swapped
    # between rigs, so never assume the saved ids are the ones on the subject.
    # Offer everything currently in range, plus whatever is saved, and say
    # plainly when a saved id is not actually present.
    in_range = [f['id'] for f in found]
    opts = list(dict.fromkeys(in_range + [i for i in (ids['left'], ids['right']) if i]))
    if found:
        stale = [f"{sd}={ids[sd]}" for sd in ('left', 'right')
                 if ids[sd] and ids[sd] not in in_range]
        if stale:
            st.warning(f"Saved but NOT in range: {', '.join(stale)}. "
                       f"Either that strap is off a body, or a different one is "
                       f"on the subject — pick from the scanned list below.")
        extra = [i for i in in_range if i not in (ids['left'], ids['right'])]
        if extra:
            st.info(f"Also in range: {', '.join(extra)}. Other rigs in the lab "
                    f"show up here too — check the printed id on the strap, and "
                    f"use RSSI as a hint (the straps on your subject read "
                    f"strongest).")
    if opts:
        a1, a2, a3 = st.columns([2, 2, 1])
        li = opts.index(ids['left']) if ids['left'] in opts else 0
        ri = opts.index(ids['right']) if ids['right'] in opts else (1 if len(opts) > 1 else 0)
        rssi = {f['id']: f['rssi'] for f in found}
        def _lbl(i):
            r = rssi.get(i)
            return f"{i}   ({r} dBm)" if r is not None else f"{i}   (not in range)"
        with a1:
            new_l = st.selectbox("LEFT shank", opts, index=li, format_func=_lbl)
        with a2:
            new_r = st.selectbox("RIGHT shank", opts, index=ri, format_func=_lbl)
        with a3:
            st.write("")
            if st.button("💾 Save", width="stretch"):
                if new_l == new_r:
                    st.error("Left and right cannot be the same strap.")
                else:
                    _save_sensor_ids({'left': new_l, 'right': new_r})
                    st.success("Saved. Restart the streams to apply.")
                    st.rerun()
    st.caption(f"Currently saved:  left = **{ids['left'] or '—'}**   "
               f"right = **{ids['right'] or '—'}**")

    st.markdown("---")

    # ---- streams ----
    st.subheader("2 · Streams")
    connect_to_lsl()
    s1, s2 = st.columns(2)
    for col, side in ((s1, 'left'), (s2, 'right')):
        with col:
            live = st.session_state.get(f'lsl_inlet_{side}') is not None
            run = _streamer_running(side)
            st.markdown(f"**{side.upper()}**  ·  `{ids[side] or '—'}`")
            if live:
                st.success("streaming")
            elif run:
                st.warning("connecting ...")
            else:
                st.error("not running")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("▶ Start", key=f"go_{side}", width="stretch",
                             disabled=run):
                    _start_streamer(side); time.sleep(2); st.rerun()
            with b2:
                if st.button("■ Stop", key=f"no_{side}", width="stretch",
                             disabled=not run):
                    _stop_streamer(side); st.rerun()

    st.markdown("---")

    # ---- live shake test ----
    st.subheader("3 · Confirm which leg is which")
    st.caption("Shake **one leg at a time** and watch which bar responds. If the "
               "wrong bar moves, the straps are swapped.")
    if st.button("🔄 Swap left ↔ right and restart streams"):
        _save_sensor_ids({'left': ids['right'], 'right': ids['left']})
        for sd in ('left', 'right'):
            _stop_streamer(sd)
        st.session_state.lsl_inlet_left = None
        st.session_state.lsl_inlet_right = None
        st.success("Swapped. Press Start on both streams again.")
        st.rerun()

    @st.fragment(run_every=0.8)
    def _live_bars():
        if (st.session_state.get('lsl_inlet_left') is None or
                st.session_state.get('lsl_inlet_right') is None):
            st.info("Both streams must be running for the live test.")
            return
        mv = _motion_both()
        ml = mv.get('left') or 0.0
        mr = mv.get('right') or 0.0
        top = max(ml, mr, 1.0)
        f1, f2 = st.columns(2)
        for col, name, val in ((f1, 'LEFT', ml), (f2, 'RIGHT', mr)):
            with col:
                moving = val > 2.5 * min(ml, mr) and val > 60
                st.markdown(f"**{name}** {'🟢 MOVING' if moving else ''}")
                st.progress(min(val / top, 1.0))
                st.caption(f"motion {val:.0f}")
    _live_bars()

    st.markdown("---")
    st.subheader("4 · Preflight")
    if st.button("🚦 Run preflight"):
        import subprocess
        r = subprocess.run([sys.executable, str(REPO_ROOT/'apps'/'preflight.py')],
                           capture_output=True, text=True, cwd=str(REPO_ROOT))
        (st.success if r.returncode == 0 else st.error)(
            "READY" if r.returncode == 0 else "NOT READY")
        st.code(r.stdout or r.stderr, language=None)


# ===========================================================================
# NAVIGATION ENTRY POINT
# ===========================================================================

st.set_page_config(page_title="HITLO Symmetry Console",
                   page_icon="🦾", layout="wide")
inject_theme()

_pages = [
    st.Page(page_sensors, title="Sensors", icon="🛰️"),
    st.Page(page_setup, title="Setup", icon="⚙️"),
    st.Page(page_run, title="Run Experiment", icon="🏃"),
    st.Page(page_gp_viewer, title="GP Viewer", icon="🧠"),
    st.Page(page_results, title="Results", icon="📊"),
]
nav = st.navigation(_pages, position="sidebar")
nav.run()