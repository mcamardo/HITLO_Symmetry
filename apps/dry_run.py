#!/usr/bin/env python3.12
"""
apps/dry_run.py — walk through a whole optimization without hardware.

    ./apps/dry_run.py                       # default: one simulated session
    ./apps/dry_run.py --repeats 20          # how reliably does it find the optimum
    ./apps/dry_run.py --response sweet_spot --noise 3
    ./apps/dry_run.py --replay P012         # use SI values measured on a real subject

Runs the REAL optimizer -- the same HIL_Exo, index table, ramp and acquisition
the console uses -- against a simulated participant. Nothing is mocked except
the walking.

Two things it is for:

1. Practice. See what the console will ask you to set, in what order, and how
   the suggestions move once BO takes over, before a participant is standing on
   the treadmill.

2. Sanity. A simulated subject has a known optimum, so you can ask whether the
   optimizer actually finds it, how often, and how much of the budget it spends
   getting there. That question cannot be answered from real sessions, where
   the true optimum is exactly what you do not know.

The response models are guesses about how this device affects gait. They are
NOT validated -- treat conclusions as being about the optimizer's behaviour,
not about the exoskeleton.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import yaml


# --------------------------------------------------------------- subject ----

def make_subject(kind: str, baseline: float, gain: float, noise: float,
                 rng: np.random.Generator):
    """A pretend participant: maps index x to a symmetry index.

    Every model is monotonic in torque somewhere and flattens at the extremes,
    which is the shape the device is expected to produce. The differences are
    in how much structure the GP has to find.
    """
    if kind == "linear":
        f = lambda x: baseline + gain * x
        doc = f"SI = {baseline:+.1f} {gain:+.1f}·x   (straight line)"
    elif kind == "saturating":
        f = lambda x: baseline + gain * np.tanh(1.8 * x)
        doc = f"SI = {baseline:+.1f} {gain:+.1f}·tanh(1.8x)   (flattens at the ends)"
    elif kind == "sweet_spot":
        # Effective in a band, falling off past it -- the case where a ramp that
        # only samples near zero would miss the useful region entirely.
        f = lambda x: baseline + gain * np.exp(-((x - 0.45) ** 2) / 0.18) * np.sign(1)
        doc = f"SI peaks near x=+0.45 and falls off either side"
    elif kind == "deadzone":
        # Nothing happens until the band engages -- slack at low stiffness.
        f = lambda x: baseline + gain * np.sign(x) * np.clip(abs(x) - 0.3, 0, None) / 0.7
        doc = f"no effect for |x| < 0.3, then linear   (band slack)"
    elif kind == "table":
        # Drive the response off the table's own dose_signed_Nm rather than an
        # invented shape. This is the only model tied to the actual device: it
        # carries the real ~17x asymmetry between the arms, where the whole
        # dorsiflexor side can apply at most ~2.7 Nm against the
        # plantarflexor side's ~47.
        import pandas as pd
        from pathlib import Path as _P
        _df = pd.read_csv(_P(__file__).resolve().parent.parent /
                          "config" / "index_unified.csv").sort_values("x")
        _x, _d = _df["x"].to_numpy(), _df["dose_signed_Nm"].to_numpy()
        _scale = gain / max(abs(_d).max(), 1e-9)
        f = lambda x: baseline + _scale * np.interp(x, _x, _d)
        doc = (f"SI = {baseline:+.1f} + dose(x) scaled so full PF = {gain:+.1f}%"
               f"   (real table dose, DF arm reaches only "
               f"{abs(_d[_d < 0]).max() / abs(_d).max() * gain:.1f}%)")
    else:
        raise ValueError(f"unknown response model: {kind}")

    def walk(x):
        return float(f(x) + rng.normal(0, noise))
    return walk, f, doc


def _replay_subject(subject_dir: Path, rng, noise):
    """Interpolate a response from SI values actually measured on a subject.

    Honest about what this is: those trials were not a controlled sweep of x,
    so this is a curve drawn through whatever settings happened to be used. It
    is more realistic than an analytic model in its noise and less trustworthy
    in its shape.
    """
    raise SystemExit(
        "--replay needs per-trial index values, and this session's BO state was "
        "never saved (no derivatives/hil_optimization). Use an analytic response "
        "model instead: --response linear|saturating|sweet_spot|deadzone")


# ------------------------------------------------------------------ run ----

def run_session(cfg, walk, truth, rng, verbose=True):
    """One simulated session.

    Runs with the working directory moved to a scratch dir: HIL_toolkit writes
    an autoiter_<n>/ checkpoint folder per BO iteration into the cwd, and a dry
    run would otherwise scatter dozens of them through the repo.
    """
    from hitlo.hil_exo import HIL_Exo

    class _Stub:
        cost_name = "simulated symmetry index"
        def __init__(self): self.si_target = float(cfg["Cost"].get("si_target", 0.0))

    hil = HIL_Exo(cfg, _Stub())
    hil.si_target = float(cfg["Cost"].get("si_target", 0.0))
    hil._generate_initial_parameters()
    _prev_cwd = os.getcwd()
    _scratch = tempfile.mkdtemp(prefix="hitlo_dryrun_")
    os.chdir(_scratch)

    n_steps = int(cfg["Optimization"]["n_steps"])
    target = float(cfg["Cost"].get("si_target", 0.0))
    hist = []

    for _ in range(n_steps):
        trial = hil.n + 1
        x = float(hil.x[hil.n, 0])
        row = hil.table.row(x)
        si = walk(x)
        cost = abs(si - target)

        if len(hil.x_opt) < 1:
            hil.x_opt = np.array([[x]]); hil.y_opt = np.array([si])
        else:
            hil.x_opt = np.concatenate((hil.x_opt, [[x]]))
            hil.y_opt = np.concatenate((hil.y_opt, [si]))

        phase = "ramp" if trial <= hil.n_ramp else "BO"
        hist.append(dict(trial=trial, x=x, si=si, cost=cost, phase=phase))
        if verbose:
            dev = {k: row[k] for k in ("R", "theta", "L0", "attachment_ratio")
                   if k in row and row[k] is not None}
            setting = "  ".join(f"{k}={float(v):.3f}" for k, v in dev.items())
            print(f"  {trial:2d} [{phase:4s}]  x={x:+.3f}   {setting}")
            print(f"                 -> SI {si:+6.2f}%   |SI-target| {cost:5.2f}")
        hil.n += 1
        # Advance exactly as the console does: ask BO for a continuous
        # suggestion, then snap it to a real table row via the acquisition.
        if hil.n_ramp <= hil.n < n_steps:
            try:
                if cfg["Optimization"].get("normalize", True):
                    raw = hil.BO.run(
                        hil._normalize_x(hil.x_opt).reshape(len(hil.x_opt), -1),
                        hil._mean_normalize_y(hil.y_opt).reshape(len(hil.x_opt), 1))
                    raw = float(hil._denormalize_x(raw).ravel()[0])
                else:
                    y_for_bo = -np.abs(hil.y_opt - target)
                    raw = hil.BO.run(hil.x_opt.reshape(len(hil.x_opt), -1),
                                     y_for_bo.reshape(len(hil.y_opt), 1))
                    raw = float(np.asarray(raw).ravel()[0])
                nxt = hil._next_x_from_table(raw)
            except Exception as e:
                if verbose:
                    print(f"                 (BO fell back to the table edge: {e})")
                nxt = float(hil.table.x_values[0])
            hil.x = np.concatenate((hil.x, [[nxt]]), axis=0)
    os.chdir(_prev_cwd)
    import shutil
    shutil.rmtree(_scratch, ignore_errors=True)
    return hil, hist


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--response", default="saturating",
                    choices=["table", "linear", "saturating", "sweet_spot",
                             "deadzone"])
    ap.add_argument("--baseline", type=float, default=-5.0,
                    help="simulated subject's SI at zero torque (%%)")
    ap.add_argument("--gain", type=float, default=14.0,
                    help="how strongly the device moves SI (%%)")
    ap.add_argument("--noise", type=float, default=1.5,
                    help="trial-to-trial SI noise, 1 SD (%%). Real trials here "
                         "show SEM ~0.8-1.7")
    ap.add_argument("--repeats", type=int, default=1,
                    help="run N sessions and summarise, instead of printing one")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay", default=None, help="(not available -- see --help)")
    a = ap.parse_args()

    cfg_path = REPO / "config" / "exo_symmetry_config.yml"
    if not cfg_path.exists():
        cfg_path = REPO / "config" / "exo_symmetry_config.example.yml"
    cfg = yaml.safe_load(cfg_path.read_text())
    target = float(cfg["Cost"].get("si_target", 0.0))

    if a.replay:
        _replay_subject(Path(a.replay), None, None)

    rng = np.random.default_rng(a.seed)
    walk, truth, doc = make_subject(a.response, a.baseline, a.gain, a.noise, rng)

    from hitlo.index_unified import IndexTable
    table = IndexTable(REPO / cfg["Optimization"]["index_csv"])
    grid = table.x_values
    best_x = float(grid[int(np.argmin(np.abs(truth(grid) - target)))])
    best_si = float(truth(best_x))

    print("=" * 74)
    print("  DRY RUN — real optimizer, simulated participant")
    print("=" * 74)
    print(f"  config      {cfg_path.name}")
    print(f"  response    {doc}")
    print(f"  noise       {a.noise:.1f}% SD per trial")
    print(f"  target      SI = {target:+.1f}%")
    print(f"  trials      {cfg['Optimization']['n_steps']} "
          f"({cfg['Optimization'].get('manual_ramp_trials', 5)} ramp + rest BO)")
    print(f"  best reachable  x = {best_x:+.3f}  ->  SI {best_si:+.2f}%  "
          f"(|error| {abs(best_si - target):.2f})")
    print("=" * 74)

    if a.repeats == 1:
        print()
        hil, hist = run_session(cfg, walk, truth, rng)
        post = hil.posterior_best()
        chosen = float(post.get("x", np.nan))
        print("\n" + "-" * 74)
        print(f"  optimizer settled on   x = {chosen:+.3f}   "
              f"(true best x = {best_x:+.3f})")
        print(f"  SI there would be      {truth(chosen):+.2f}%   "
              f"(target {target:+.1f}%, best achievable {best_si:+.2f}%)")
        print(f"  cost of that choice    {abs(truth(chosen) - target):.2f} "
              f"vs {abs(best_si - target):.2f} at the true optimum")
        ramp = [h for h in hist if h["phase"] == "ramp"]
        print(f"\n  ramp covered x from {min(h['x'] for h in ramp):+.2f} to "
              f"{max(h['x'] for h in ramp):+.2f}")
        print(f"  BO trials explored  {min(h['x'] for h in hist if h['phase']=='BO'):+.2f} "
              f"to {max(h['x'] for h in hist if h['phase']=='BO'):+.2f}")
    else:
        errs, chosens = [], []
        for i in range(a.repeats):
            r2 = np.random.default_rng(a.seed + i)
            w2, _, _ = make_subject(a.response, a.baseline, a.gain, a.noise, r2)
            hil, _ = run_session(cfg, w2, truth, r2, verbose=False)
            c = float(hil.posterior_best().get("x", np.nan))
            chosens.append(c); errs.append(abs(truth(c) - target))
        errs = np.array(errs); chosens = np.array(chosens)
        floor = abs(best_si - target)
        print(f"\n  {a.repeats} simulated sessions:")
        print(f"    chosen x       median {np.median(chosens):+.3f}   "
              f"IQR [{np.percentile(chosens,25):+.3f}, {np.percentile(chosens,75):+.3f}]")
        print(f"    |SI - target|  median {np.median(errs):.2f}   "
              f"worst {errs.max():.2f}   (floor {floor:.2f})")
        within = float(np.mean(errs <= floor + 1.0) * 100)
        print(f"    within 1 point of the best reachable: {within:.0f}% of sessions")
        if within < 70:
            print(f"\n  Under 70%. With {a.noise:.1f}% noise and "
                  f"{cfg['Optimization']['n_steps']} trials that may simply be the "
                  f"budget, not a fault -- try --noise 0 to see the ceiling.")
    print()
    print("  The response models are guesses about how this device affects gait,")
    print("  and are not validated. Read the numbers as being about the")
    print("  optimizer's behaviour, not about the exoskeleton.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
