"""
apps/gp_viewer.py — Interactive Streamlit viewer for BO iteration progress.

Loads each GP model checkpoint saved during an experiment and renders:
  1. GP mean surface (predicted cost landscape)
  2. GP uncertainty surface (where the model is unsure)
  3. Acquisition function surface (where BO thinks to sample next)

Lets you step iteration-by-iteration through the optimization to see how the
model's belief evolved as data came in.

Usage
-----
    streamlit run apps/gp_viewer.py -- --subject P049 --session S001

Or edit the SETTINGS at the top of this file.

The GP models the Bayesian Optimization cost function, which is the signed
symmetry magnitude |SI|. BO MINIMIZES this toward zero.

Display modes:
  - "Predicted |Asymmetry %|" — intuitive back-transform showing predicted
    gait asymmetry magnitude across the (R, L₀) space. Zero = perfect symmetry.
  - "BO Score (Internal)"     — raw normalized/negated objective the BO sees.
    Higher = better because BoTorch maximizes.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from plotly.subplots import make_subplots


# ===========================================================================
# SETTINGS — override via --subject / --session on the streamlit command line
# ===========================================================================

SUBJECT = 'P042'
SESSION = 'S001'
BASE_DIR = '/Users/maccamardo/HITLO'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--subject', default=SUBJECT)
    p.add_argument('--session', default=SESSION)
    p.add_argument('--base-dir', default=BASE_DIR)
    # Streamlit passes its own args; ignore anything we don't know
    args, _ = p.parse_known_args()
    return args


args = parse_args()


# ===========================================================================
# Parameter bounds — read from config if available, else fall back
# ===========================================================================

def load_param_range():
    REPO_ROOT = Path(__file__).resolve().parent.parent
    for candidate in [
        REPO_ROOT / 'config' / 'exo_symmetry_config.yml',
        Path('config/exo_symmetry_config.yml'),
    ]:
        if candidate.exists():
            with open(candidate) as f:
                cfg = yaml.safe_load(f)
            rng = np.array(list(cfg['Optimization']['range'])).reshape(2, 2)
            return rng[0, 0], rng[1, 0], rng[0, 1], rng[1, 1]
    # Fallback defaults (match the old script)
    return 0.24, 0.35, 0.30, 0.40


R_MIN, R_MAX, L0_MIN, L0_MAX = load_param_range()


# ===========================================================================
# Page layout
# ===========================================================================

st.set_page_config(page_title="HITLO_Symmetry GP Viewer", layout="wide")

st.title("🎯 HITLO_Symmetry — GP Optimization Viewer")
st.markdown(f"**Subject:** {args.subject}    **Session:** {args.session}    "
            f"Bounds: R [{R_MIN:.3f}–{R_MAX:.3f}]  L₀ [{L0_MIN:.3f}–{L0_MAX:.3f}]")

st.info(
    "**What you're looking at:** the Gaussian Process learns the cost "
    "function |symmetry| across (R, L₀) space. The BO picks new trials by "
    "maximizing an acquisition function (qNoisyEI) that balances "
    "exploitation (predicted low cost) vs exploration (high uncertainty)."
)


# ===========================================================================
# Load session data
# ===========================================================================

models_dir = Path(f'{args.base_dir}/sub-{args.subject}/ses-{args.session}/'
                  f'derivatives/hil_optimization/models')
hil_csv = Path(f'{args.base_dir}/sub-{args.subject}/ses-{args.session}/eeg/'
               f'sub-{args.subject}_ses-{args.session}_hil_results.csv')

if not hil_csv.exists():
    st.error(f"❌ HIL results not found: {hil_csv}")
    st.info("This viewer loads data produced by a completed experiment. "
            "If you haven't run a session yet (or the subject/session name "
            "is wrong), there's nothing to view.")
    st.stop()

hil_results = pd.read_csv(hil_csv)
st.sidebar.info(f"✅ {len(hil_results)} trials in hil_results.csv")

iter_folders = sorted(
    [f for f in models_dir.glob('iter_*') if f.is_dir()],
    key=lambda x: int(x.name.split('_')[1])
)
if not iter_folders:
    st.error(f"❌ No iteration folders in {models_dir}")
    st.stop()

iter_numbers = [int(f.name.split('_')[1]) for f in iter_folders]
st.sidebar.success(
    f"✅ {len(iter_numbers)} GP checkpoints  "
    f"(iter {min(iter_numbers)}–{max(iter_numbers)})")


# ===========================================================================
# Load & reconstruct GP for a given iteration
# ===========================================================================

@st.cache_data
def load_gp_data(iter_path_str, iter_num):
    iter_path = Path(iter_path_str)
    csv_path = iter_path / 'data.csv'
    model_path = iter_path / 'model.pth'
    if not csv_path.exists() or not model_path.exists():
        return None

    try:
        data = np.loadtxt(csv_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        R_norm, L0_norm, bo_cost = data[:, 0], data[:, 1], data[:, 2]

        # Denormalize to physical units
        R = R_norm * (R_MAX - R_MIN) + R_MIN
        L0 = L0_norm * (L0_MAX - L0_MIN) + L0_MIN

        actual_asymmetry = hil_results['cost'].values[:iter_num]

        # Prediction grid
        R_grid_vals = np.linspace(R_MIN, R_MAX, 40)
        L0_grid_vals = np.linspace(L0_MIN, L0_MAX, 40)
        R_grid, L0_grid = np.meshgrid(R_grid_vals, L0_grid_vals)
        R_g_norm = (R_grid - R_MIN) / (R_MAX - R_MIN)
        L0_g_norm = (L0_grid - L0_MIN) / (L0_MAX - L0_MIN)
        X_test = np.column_stack([R_g_norm.ravel(), L0_g_norm.ravel()])

        # Reconstruct GP model
        X_train = torch.tensor(np.column_stack([R_norm, L0_norm]),
                               dtype=torch.float64)
        y_train = torch.tensor(bo_cost, dtype=torch.float64).reshape(-1, 1)
        likelihood = GaussianLikelihood()
        model = SingleTaskGP(X_train, y_train, likelihood=likelihood)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        with torch.no_grad():
            X_test_torch = torch.tensor(X_test, dtype=torch.float64)
            pred = likelihood(model(X_test_torch))
            mean_bo = pred.mean.numpy()
            lower_bo, upper_bo = pred.confidence_region()
            lower_bo = lower_bo.numpy()
            upper_bo = upper_bo.numpy()

        # Back-transform BO score → |asymmetry %|
        abs_a = np.abs(actual_asymmetry)
        mean_a = abs_a.mean()
        std_a = abs_a.std() if abs_a.std() > 0 else 1.0
        predicted_asymm = np.clip((-mean_bo * std_a) + mean_a, 0, None)
        predicted_lower = np.clip((-upper_bo * std_a) + mean_a, 0, None)
        predicted_upper = np.clip((-lower_bo * std_a) + mean_a, 0, None)

        # Acquisition function
        try:
            from botorch.acquisition import qNoisyExpectedImprovement
            from botorch.sampling import IIDNormalSampler

            sampler = IIDNormalSampler(
                sample_shape=torch.Size([200]), seed=1234)
            acq = qNoisyExpectedImprovement(model, X_train, sampler=sampler)
            acq_vals = []
            for start in range(0, len(X_test), 100):
                end = min(start + 100, len(X_test))
                x_batch = torch.tensor(
                    X_test[start:end], dtype=torch.float64).unsqueeze(1)
                with torch.no_grad():
                    acq_vals.append(acq(x_batch).numpy())
            acq_grid = np.concatenate(acq_vals).reshape(R_grid.shape)
        except Exception:
            acq_grid = np.zeros(R_grid.shape)

        n_min = min(len(R), len(L0), len(bo_cost), len(actual_asymmetry))
        return {
            'R_grid': R_grid, 'L0_grid': L0_grid,
            'mean_grid_bo': mean_bo.reshape(R_grid.shape),
            'lower_grid_bo': lower_bo.reshape(R_grid.shape),
            'upper_grid_bo': upper_bo.reshape(R_grid.shape),
            'mean_grid_asymm': predicted_asymm.reshape(R_grid.shape),
            'lower_grid_asymm': predicted_lower.reshape(R_grid.shape),
            'upper_grid_asymm': predicted_upper.reshape(R_grid.shape),
            'R_points': R[:n_min],
            'L0_points': L0[:n_min],
            'bo_cost_points': bo_cost[:n_min],
            'asymmetry_points': actual_asymmetry[:n_min],
            'iteration': iter_num,
            'n_points': n_min,
            'acq_grid': acq_grid,
        }

    except Exception as e:
        st.error(f"Error loading iteration {iter_num}: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


# ===========================================================================
# Sidebar controls
# ===========================================================================

st.sidebar.header("Controls")

display_mode = st.sidebar.radio(
    "Display mode",
    options=["Predicted |Asymmetry %|", "BO Score (internal)"],
    help=(
        "'Predicted |Asymmetry %|' back-transforms the GP output to a "
        "gait asymmetry estimate (zero = symmetric). "
        "'BO Score' shows the raw objective (higher = better to BO)."
    ),
)

selected_iter = st.sidebar.select_slider(
    "Iteration", options=iter_numbers, value=iter_numbers[0])

gp_data = load_gp_data(str(models_dir / f'iter_{selected_iter}'), selected_iter)
if gp_data is None:
    st.error(f"❌ Could not load iteration {selected_iter}")
    st.stop()


# ===========================================================================
# Top metrics
# ===========================================================================

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Iteration", selected_iter)
with c2:
    st.metric("Total trials so far", gp_data['n_points'])
with c3:
    best_abs = np.abs(gp_data['asymmetry_points']).min()
    st.metric("|Best SI|", f"{best_abs:.2f}%")
with c4:
    best_idx = np.argmin(np.abs(gp_data['asymmetry_points']))
    st.metric(
        "Best params",
        f"R={gp_data['R_points'][best_idx]:.3f}, "
        f"L₀={gp_data['L0_points'][best_idx]:.3f}",
    )


# ===========================================================================
# Select the surface data
# ===========================================================================

if display_mode.startswith("Predicted"):
    mean_grid = gp_data['mean_grid_asymm']
    lower_grid = gp_data['lower_grid_asymm']
    upper_grid = gp_data['upper_grid_asymm']
    cs = 'RdBu'
    cb_title = 'Predicted<br>|SI %|'
    subtitle = 'GP Predicted |SI %|  (zero = symmetric)'
    z_label = '|SI %|'
else:
    mean_grid = gp_data['mean_grid_bo']
    lower_grid = gp_data['lower_grid_bo']
    upper_grid = gp_data['upper_grid_bo']
    cs = 'RdYlGn'
    cb_title = 'BO Score<br>(higher=better)'
    subtitle = 'GP Mean: BO Score (BoTorch maximizes)'
    z_label = 'BO Score'


# ===========================================================================
# Three-panel figure
# ===========================================================================

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(subtitle, 'GP Uncertainty',
                    'Acquisition Function (where to sample next)'),
    specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
    horizontal_spacing=0.05,
)

# --- Panel 1: GP mean ---
fig.add_trace(go.Surface(
    x=gp_data['R_grid'], y=gp_data['L0_grid'], z=mean_grid,
    colorscale=cs, colorbar=dict(x=0.3, len=0.75, title=cb_title, thickness=12),
), row=1, col=1)

fig.add_trace(go.Scatter3d(
    x=gp_data['R_points'], y=gp_data['L0_points'],
    z=np.abs(gp_data['asymmetry_points']),
    mode='markers+text',
    marker=dict(size=8, color='red', line=dict(color='black', width=2)),
    text=[str(i + 1) for i in range(len(gp_data['R_points']))],
    textposition='top center', textfont=dict(size=10, color='white'),
    name='Measured |SI %|',
    customdata=list(zip(np.abs(gp_data['asymmetry_points']),
                        gp_data['bo_cost_points'])),
    hovertemplate=(
        '<b>Trial %{text}</b><br>'
        '|SI|: %{customdata[0]:.2f}%<br>'
        'BO: %{customdata[1]:.3f}<br>'
        'R=%{x:.3f}<br>L₀=%{y:.3f}<extra></extra>'),
), row=1, col=1)

# Trajectory path
for i in range(gp_data['n_points'] - 1):
    fig.add_trace(go.Scatter3d(
        x=[gp_data['R_points'][i], gp_data['R_points'][i + 1]],
        y=[gp_data['L0_points'][i], gp_data['L0_points'][i + 1]],
        z=[np.abs(gp_data['asymmetry_points'][i]),
           np.abs(gp_data['asymmetry_points'][i + 1])],
        mode='lines', line=dict(color='white', width=2),
        showlegend=False, hoverinfo='skip',
    ), row=1, col=1)

# --- Panel 2: uncertainty ---
uncertainty = upper_grid - lower_grid
fig.add_trace(go.Surface(
    x=gp_data['R_grid'], y=gp_data['L0_grid'], z=uncertainty,
    colorscale='Reds',
    colorbar=dict(x=0.64, len=0.75, title='Uncertainty', thickness=12),
), row=1, col=2)
fig.add_trace(go.Scatter3d(
    x=gp_data['R_points'], y=gp_data['L0_points'],
    z=np.zeros_like(gp_data['asymmetry_points']),
    mode='markers',
    marker=dict(size=8, color='blue', line=dict(color='black', width=2)),
    showlegend=False,
), row=1, col=2)

# --- Panel 3: acquisition function ---
acq_grid = gp_data['acq_grid']
best_acq_idx = np.argmax(acq_grid.ravel())
best_acq_R = gp_data['R_grid'].ravel()[best_acq_idx]
best_acq_L0 = gp_data['L0_grid'].ravel()[best_acq_idx]
best_acq_val = acq_grid.ravel()[best_acq_idx]

fig.add_trace(go.Surface(
    x=gp_data['R_grid'], y=gp_data['L0_grid'], z=acq_grid,
    colorscale='Viridis',
    colorbar=dict(x=1.0, len=0.75, title='EI', thickness=12),
), row=1, col=3)

fig.add_trace(go.Scatter3d(
    x=[best_acq_R], y=[best_acq_L0], z=[best_acq_val],
    mode='markers+text',
    marker=dict(size=12, color='yellow', symbol='diamond',
                line=dict(color='black', width=2)),
    text=['Next →'], textposition='top center',
    textfont=dict(size=11, color='yellow'),
    name=f'Suggested next (R={best_acq_R:.3f}, L₀={best_acq_L0:.3f})',
    hovertemplate=(
        '<b>Suggested next sample</b><br>'
        f'R={best_acq_R:.3f}<br>'
        f'L₀={best_acq_L0:.3f}<br>'
        f'EI={best_acq_val:.4f}<extra></extra>'),
), row=1, col=3)

fig.add_trace(go.Scatter3d(
    x=gp_data['R_points'], y=gp_data['L0_points'],
    z=np.zeros_like(gp_data['asymmetry_points']),
    mode='markers', marker=dict(size=6, color='red', opacity=0.7),
    showlegend=False,
), row=1, col=3)

cam = dict(eye=dict(x=1.5, y=1.5, z=1.3))
fig.update_layout(
    title=f'Iteration {selected_iter} — {gp_data["n_points"]} trials',
    height=620,
    scene=dict(xaxis_title='R (m)', yaxis_title='L₀ (m)',
               zaxis_title=z_label, camera=cam),
    scene2=dict(xaxis_title='R (m)', yaxis_title='L₀ (m)',
                zaxis_title='Uncertainty', camera=cam),
    scene3=dict(xaxis_title='R (m)', yaxis_title='L₀ (m)',
                zaxis_title='EI', camera=cam),
)

st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# Trial data table
# ===========================================================================

with st.expander("📊 Trial data"):
    df = pd.DataFrame({
        'Trial': range(1, gp_data['n_points'] + 1),
        'R (m)': gp_data['R_points'],
        'L₀ (m)': gp_data['L0_points'],
        'BO cost': gp_data['bo_cost_points'],
        '|SI %|': np.abs(gp_data['asymmetry_points']),
        'signed SI %': gp_data['asymmetry_points'],
    })
    st.dataframe(df, use_container_width=True)


st.sidebar.markdown("---")
st.sidebar.info(f"📂 Models:\n`{models_dir}`")
st.sidebar.info(f"📊 HIL:\n`{hil_csv.name}`")
