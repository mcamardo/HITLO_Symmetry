#!/usr/bin/env python3.12
"""
apps/trial_explorer.py — browse any recording and see what detection did to it.

    streamlit run apps/trial_explorer.py

Pick a trial from the sidebar and get the whole picture: where the walking was,
where every heel strike landed on the raw signal, what the step times and
symmetry index came out as, and how the two detectors disagree.

WHY THIS EXISTS
---------------
diagnose_trial.py answers "is this trial usable" for one file, on the terminal,
with a fixed set of panels. This is for the other question -- "what is actually
in this recording, and why did the number come out like that" -- which needs
zooming, switching detectors, and jumping between trials.

Every number here comes from the same hitlo functions the cost function uses,
so what you see is what the optimizer saw. Nothing is recomputed a second way.
"""

import glob
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hitlo.io import load_streams, sensing_config
from hitlo.detectors import detect, detector_name
from hitlo.symmetry import compute_step_times, compute_symmetry_index

RED, GRN, GREY = "#b5121b", "#4cae4f", "#9aa0a6"
INK, MUTE, GRID = "#12131a", "#6b6f76", "#e2e4e8"

st.set_page_config(page_title="Trial Explorer", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------- data ----

def _config():
    import yaml
    for name in ("exo_symmetry_config.yml", "exo_symmetry_config.example.yml"):
        p = REPO / "config" / name
        if p.exists():
            return yaml.safe_load(p.read_text()) or {}
    return {}


def _data_root(cfg):
    base = (cfg.get("Subject") or {}).get("base_dir", "~/HITLO_Data")
    return Path(os.path.expanduser(str(base)))


@st.cache_data(show_spinner=False)
def _find_trials(root: str):
    """Every .xdf under the data root, newest first, excluding _old duplicates."""
    files = [f for f in glob.glob(f"{root}/**/*.xdf", recursive=True)
             if "_old" not in os.path.basename(f)]
    return sorted(files, key=os.path.getmtime, reverse=True)


@st.cache_data(show_spinner=False)
def _load(path: str, backend: str):
    """Streams plus the walking window. Cached -- XDF parsing is the slow part."""
    L, R = load_streams(path, {"Sensing": {"backend": backend}})
    if L is None or R is None:
        return None
    t0 = float(L.timestamps[0])
    t = np.asarray(L.timestamps, float) - t0
    fs = 1.0 / float(np.median(np.diff(t)))

    # Walking window from total angular velocity (gyro) or accel variability.
    if getattr(L, "has_gyro", False):
        act = (np.linalg.norm(np.asarray(L.gyro, float), axis=1) +
               np.linalg.norm(np.asarray(R.gyro, float), axis=1)) / 2.0
        thresh = 90.0
    else:
        m = np.linalg.norm(np.asarray(L.accel, float), axis=1)
        act = np.abs(m - np.median(m))
        thresh = max(0.25, float(np.percentile(act, 60)) * 0.6)
    k = max(int(fs), 1)
    act = np.convolve(act, np.ones(k) / k, "same")

    idx = np.flatnonzero(act > thresh)
    if len(idx) < fs:
        W0, W1 = float(t[0]), float(t[-1])
    else:
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > k) + 1)
        r = max(runs, key=len)
        W0, W1 = float(t[r[0]]), float(t[r[-1]])
    return dict(L=L, R=R, t0=t0, t=t, fs=fs, act=act, thresh=thresh, W0=W0, W1=W1)


def _events(stream, t0, W0, W1, method, config):
    kw = {} if method is None else {"method": method}
    res = detect(stream, config, **kw)
    v = np.asarray(res.heel_strike_times, float) - t0
    return np.sort(v[(v >= W0) & (v <= W1)]), res


def _si(evL, evR):
    rs, ls = compute_step_times(evL, evR)
    n = min(len(rs), len(ls))
    if n < 4:
        return None
    si, per = compute_symmetry_index(rs[:n], ls[:n], signed=True)
    return dict(si=si, per=per, n=n, rs=rs[:n], ls=ls[:n],
                sd=float(np.std(per, ddof=1)),
                sem=float(np.std(per, ddof=1) / np.sqrt(n)))


# ------------------------------------------------------------- sidebar ----

cfg = _config()
root = _data_root(cfg)
st.sidebar.title("🔎 Trial Explorer")

if not root.exists():
    st.error(f"Data root does not exist: `{root}`\n\n"
             f"Set `Subject.base_dir` in `config/exo_symmetry_config.yml`.")
    st.stop()

trials = _find_trials(str(root))
if not trials:
    st.error(f"No .xdf recordings found under `{root}`.")
    st.stop()


def _label(p):
    rel = os.path.relpath(p, root)
    mt = os.path.getmtime(p)
    import datetime
    return f"{rel}   ({datetime.datetime.fromtimestamp(mt):%b %d %H:%M})"


choice = st.sidebar.selectbox("Recording", trials, format_func=_label)
backend = st.sidebar.radio(
    "Backend", ["trigno", "polar"],
    index=0 if sensing_config(cfg).get("backend", "trigno") == "trigno" else 1,
    help="How to demultiplex the file. Trigno files carry both sides in one "
         "stream; Polar files have one stream per side.")

data = _load(choice, backend)
if data is None:
    st.error("Could not load left/right streams from this file with the "
             f"**{backend}** backend. Try the other one.")
    st.stop()

L, R, t, fs = data["L"], data["R"], data["t"], data["fs"]
has_gyro = bool(getattr(L, "has_gyro", False))

st.sidebar.markdown("---")
methods = ["gyro", "accel"] if has_gyro else ["accel"]
if not has_gyro:
    st.sidebar.info("No gyroscope in this file — accelerometer only.")
show = st.sidebar.multiselect("Detectors", methods, default=methods)

st.sidebar.markdown("---")
st.sidebar.caption("Analysis window")
use_auto = st.sidebar.checkbox("Auto-detected walking segment", value=True)
if use_auto:
    W0, W1 = data["W0"], data["W1"]
else:
    W0, W1 = st.sidebar.slider("Window (s)", 0.0, float(t[-1]),
                               (float(data["W0"]), float(data["W1"])), step=1.0)
trim = st.sidebar.number_input("Extra trim each end (s)", 0.0, 20.0, 0.0, 0.5)
W0, W1 = W0 + trim, W1 - trim

# ---------------------------------------------------------------- main ----

st.title(os.path.basename(choice))
st.caption(f"{t[-1]:.0f} s recorded · {fs:.0f} Hz · walking {W0:.0f}–{W1:.0f} s "
           f"({W1 - W0:.0f} s) · backend **{backend}**"
           + (" · gyro present" if has_gyro else " · accelerometer only"))

EV, SI = {}, {}
for m in show:
    try:
        EV[m] = {s: _events(S, data["t0"], W0, W1, m, cfg)[0]
                 for s, S in (("L", L), ("R", R))}
        SI[m] = _si(EV[m]["L"], EV[m]["R"])
    except Exception as e:
        st.warning(f"`{m}` detector failed on this file: {e}")

if not EV:
    st.stop()

cols = st.columns(len(show) + 1)
for i, m in enumerate(show):
    r = SI.get(m)
    with cols[i]:
        if r is None:
            st.metric(f"SI · {m}", "—", "too few strides")
        else:
            st.metric(f"SI · {m}", f"{r['si']:+.2f}%",
                      f"±{r['sem']:.2f} SEM · n={r['n']}")
with cols[-1]:
    if len(SI) == 2 and all(v is not None for v in SI.values()):
        a, b = [SI[m]["si"] for m in show]
        st.metric("Detectors differ by", f"{abs(a - b):.2f} pts",
                  "agree" if abs(a - b) < 3 else "investigate",
                  delta_color="normal" if abs(a - b) < 3 else "inverse")

st.markdown("---")

# ---- activity + signal -----------------------------------------------------
st.subheader("Signal and detected heel strikes")
what = st.radio("Show", ["Sagittal gyro" if has_gyro else "Acceleration |a|",
                         "Acceleration |a|"] if has_gyro else ["Acceleration |a|"],
                horizontal=True, label_visibility="collapsed")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.22, 0.39, 0.39], vertical_spacing=0.06,
                    subplot_titles=("Activity — shaded band is the analysis window",
                                    "LEFT", "RIGHT"))
fig.add_trace(go.Scatter(x=t, y=data["act"], line=dict(color=GREY, width=1),
                         fill="tozeroy", name="activity", showlegend=False), row=1, col=1)
fig.add_vrect(x0=W0, x1=W1, fillcolor=GRN, opacity=0.13, line_width=0, row=1, col=1)

for row, (side, S, c) in enumerate((("L", L, RED), ("R", R, GRN)), start=2):
    if what.startswith("Sagittal") and has_gyro:
        res = detect(S, cfg, method="gyro")
        y = np.asarray(res.magnitude, float)
        yname = "deg/s"
    else:
        y = np.linalg.norm(np.asarray(S.accel, float), axis=1)
        yname = "g"
    fig.add_trace(go.Scatter(x=t, y=y, line=dict(color=c, width=1.2),
                             name=f"{side}", showlegend=False), row=row, col=1)
    for m, sym, mc in (("gyro", "circle-open", INK), ("accel", "x", INK)):
        if m not in EV:
            continue
        e = EV[m][side]
        fig.add_trace(go.Scatter(
            x=e, y=np.interp(e, t, y), mode="markers",
            marker=dict(symbol=sym, size=9, color=mc,
                        line=dict(width=2, color=mc)),
            name=f"{m} · {side}", showlegend=(row == 2)), row=row, col=1)
    fig.update_yaxes(title_text=yname, row=row, col=1)

fig.update_layout(height=680, margin=dict(l=60, r=20, t=50, b=40),
                  plot_bgcolor="white", paper_bgcolor="white",
                  legend=dict(orientation="h", y=1.08, x=0),
                  hovermode="x unified")
fig.update_xaxes(title_text="time (s)", row=3, col=1, gridcolor=GRID)
fig.update_yaxes(gridcolor=GRID)
st.plotly_chart(fig, use_container_width=True)
st.caption("Drag to zoom, double-click to reset. ○ gyro contact · ✕ accelerometer impact")

# ---- step times / SI -------------------------------------------------------
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.subheader("Step times")
    f2 = go.Figure()
    for m, dash in (("gyro", "solid"), ("accel", "dash")):
        if m not in SI or SI[m] is None:
            continue
        f2.add_trace(go.Scatter(y=SI[m]["rs"], name=f"right · {m}",
                                line=dict(color=GRN, dash=dash, width=2)))
        f2.add_trace(go.Scatter(y=SI[m]["ls"], name=f"left · {m}",
                                line=dict(color=RED, dash=dash, width=2)))
    f2.update_layout(height=340, margin=dict(l=50, r=20, t=20, b=40),
                     plot_bgcolor="white", xaxis_title="stride",
                     yaxis_title="step time (s)",
                     legend=dict(orientation="h", y=1.12, x=0))
    f2.update_xaxes(gridcolor=GRID); f2.update_yaxes(gridcolor=GRID)
    st.plotly_chart(f2, use_container_width=True)

with c2:
    st.subheader("Symmetry index, stride by stride")
    f3 = go.Figure()
    for m, c, dash in (("gyro", GRN, "solid"), ("accel", GREY, "dash")):
        if m not in SI or SI[m] is None:
            continue
        f3.add_trace(go.Scatter(y=SI[m]["per"], name=f"{m}  ({SI[m]['si']:+.2f}%)",
                                mode="lines+markers",
                                line=dict(color=c, dash=dash, width=2),
                                marker=dict(size=5)))
        f3.add_hline(y=SI[m]["si"], line=dict(color=c, dash="dot", width=1.5))
    f3.add_hline(y=0, line=dict(color=INK, width=1.5))
    f3.update_layout(height=340, margin=dict(l=50, r=20, t=20, b=40),
                     plot_bgcolor="white", xaxis_title="stride",
                     yaxis_title="per-stride SI (%)",
                     legend=dict(orientation="h", y=1.12, x=0))
    f3.update_xaxes(gridcolor=GRID); f3.update_yaxes(gridcolor=GRID)
    st.plotly_chart(f3, use_container_width=True)

# ---- quality ---------------------------------------------------------------
st.markdown("---")
st.subheader("Detection quality")
rows = []
for m in show:
    r = SI.get(m)
    if r is None:
        rows.append({"detector": m, "note": "too few strides"})
        continue
    st_all = np.concatenate([r["rs"], r["ls"]])
    ivL, ivR = np.diff(EV[m]["L"]), np.diff(EV[m]["R"])
    stride = float(np.median(np.concatenate([ivL, ivR]))) if len(ivL) and len(ivR) else np.nan
    rows.append({
        "detector": m,
        "strides L / R": f"{len(EV[m]['L'])} / {len(EV[m]['R'])}",
        "SI (%)": round(r["si"], 2),
        "per-stride sd": round(r["sd"], 2),
        "stride (s)": round(stride, 3),
        "stride sd (s)": round(float(np.std(np.concatenate([ivL, ivR]))), 3),
        "impossible steps": int(np.sum((st_all < 0.45) | (st_all > 1.10))),
        "|SI| > 25": int(np.sum(np.abs(r["per"]) > 25)),
    })
st.dataframe(rows, use_container_width=True, hide_index=True)

if len(EV) == 2:
    a, b = EV["gyro"], EV["accel"]
    lines = []
    for side in ("L", "R"):
        g, ac = a[side], b[side]
        o = [(g[int(np.argmin(np.abs(g - x)))] - x) * 1000 for x in ac
             if len(g) and abs(g[int(np.argmin(np.abs(g - x)))] - x) < 0.35]
        only_g = [x for x in g if len(ac) == 0 or
                  abs(ac[int(np.argmin(np.abs(ac - x)))] - x) > 0.25]
        only_a = [x for x in ac if len(g) == 0 or
                  abs(g[int(np.argmin(np.abs(g - x)))] - x) > 0.25]
        med = f"{np.median(o):+.0f} ms" if len(o) > 4 else "—"
        lines.append(f"- **{side}**: gyro leads accel by {med} "
                     f"· gyro-only events **{len(only_g)}** "
                     f"· accel-only events **{len(only_a)}**")
    st.markdown("\n".join(lines))
    st.caption("An event one detector has and the other does not is usually a "
               "missed strike, not a timing disagreement. The gyro genuinely "
               "leads the accelerometer — it marks contact, the accelerometer "
               "marks the impact that follows.")
