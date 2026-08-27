#!/usr/bin/env python3.12
"""
apps/check_calibration.py — did the ankle axis calibration take?

    ./apps/check_calibration.py                 # the newest recording
    ./apps/check_calibration.py <file.xdf>      # a specific one
    ./apps/check_calibration.py --demo          # what pass and fail look like
    ./apps/check_calibration.py --rehearse      # walk through it step by step

Run this while the participant still has the sensors on. The calibration
encodes where each sensor sits on the limb, so a failure found during offline
analysis cannot be repaired -- the session simply has no usable ankle
magnitude. Re-recording at the bench takes two minutes.

Nothing here is needed for step time or the symmetry index; those use the
shank gyroscopes only.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from hitlo.ankle_angle import (functional_calibration, find_movement_segments,
                               validate_functional_calibration,
                               ankle_angle_functional, FUNCTIONAL_CALIBRATION)

MARK = {"ok": "  ok  ", " warn": " warn ", "fail": " FAIL "}
BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"
GRN, YEL, RED = "\033[32m", "\033[33m", "\033[31m"
COLOR = {"ok": GRN, "warn": YEL, "fail": RED}


def _root():
    import yaml
    for name in ("exo_symmetry_config.yml", "exo_symmetry_config.example.yml"):
        p = REPO / "config" / name
        if p.exists():
            cfg = yaml.safe_load(p.read_text()) or {}
            base = (cfg.get("Subject") or {}).get("base_dir") or "~/HITLO_Data"
            return os.path.expanduser(str(base))
    return os.path.expanduser("~/HITLO_Data")


def _newest():
    files = [f for f in glob.glob(f"{_root()}/**/*.xdf", recursive=True)
             if "_old" not in os.path.basename(f)]
    return max(files, key=os.path.getmtime) if files else None


def report(cal, label):
    """Print the per-check table. Returns True if the calibration is usable."""
    v = validate_functional_calibration(cal)
    print(f"\n  {BOLD}{label}{OFF}")
    for c in v["checks"]:
        col = COLOR[c["level"]]
        print(f"    {col}[{c['level']:^4}]{OFF} {c['name']:<24} {c['detail']}")
        if c["level"] != "ok":
            print(f"           {DIM}{c['why']}{OFF}")
    if not v["ok"]:
        print(f"\n    {RED}The axis was not identified.{OFF} Re-record now, "
              f"while the sensors are still on.")
    elif v["warn"]:
        print(f"\n    {YEL}Usable, but check the warnings{OFF} before trusting "
              f"the sign or small between-trial differences.")
    else:
        print(f"\n    {GRN}Axis measured.{OFF} Ankle magnitude is calibrated "
              f"for this mounting.")
    return v["ok"]


def check_file(path):
    from hitlo.io import load_trigno_segment, trigno_inventory
    print(f"\n{BOLD}{os.path.basename(path)}{OFF}")
    try:
        inv = trigno_inventory(path)
    except Exception as e:
        print(f"  {RED}Could not read that recording:{OFF} {e}")
        return 1
    sides = [s for s, seg in inv.items() if "foot" in seg and "shank" in seg]
    if not sides:
        have = {s: sorted(seg) for s, seg in inv.items()}
        print(f"  {RED}No leg carries both a foot and a shank sensor.{OFF}")
        print(f"  Found: {have}")
        print(f"  The ankle axis is the angle between two segments, so one "
              f"sensor alone cannot give it.")
        return 1

    worst = 0
    for side in sides:
        foot = load_trigno_segment(path, side, "foot")
        shank = load_trigno_segment(path, side, "shank")
        try:
            cal = functional_calibration(foot, shank)
        except ValueError as e:
            seg = find_movement_segments(foot, shank)
            found = [k for k in ("foot", "shank", "rigid") if seg.get(k)]
            print(f"\n  {BOLD}{side} leg{OFF}")
            print(f"    {RED}Not a usable calibration.{OFF} "
                  f"{str(e).splitlines()[0]}")
            print(f"    Parts detected: {found if found else 'none'}")
            worst = 1
            continue
        if not report(cal, f"{side} leg"):
            worst = 1
        # a calibration is only worth anything if it produces a sane angle
        try:
            from hitlo.detectors import detect
            hs = np.asarray(detect(shank, {"Sensing": {"backend": "trigno",
                                                       "detector": "gyro"}}
                                   ).heel_strike_times, float)
            hs = np.sort(hs - float(np.asarray(shank.timestamps, float)[0]))
            res = ankle_angle_functional(foot, shank, cal,
                                         heel_strike_times=hs)
            if res.get("profile") is not None:
                print(f"    {DIM}{res['n_strides']} strides in this file · "
                      f"range of motion {res['rom']:.1f}° · peak "
                      f"plantarflexion {res['plantarflexion_peak']:+.1f}° at "
                      f"{res['plantarflexion_at']}% of stride{OFF}")
                print(f"    {DIM}(literature: 25-30° range, plantarflexion "
                      f"peak near 62%){OFF}")
        except Exception:
            pass
    return worst


def _synth(good=True, seed=5):
    """A calibration recording with a known 36 degree ankle excursion."""
    from scipy.spatial.transform import Rotation as Rot
    from hitlo.io import SensorStream
    r = np.random.default_rng(seed)
    fs, axis = 148.0, np.array([0.0, 1.0, 0.0])
    R1 = Rot.from_euler("xyz", [12, -8, 30], degrees=True)
    R2 = Rot.from_euler("xyz", [-20, 15, -40], degrees=True)
    blk = lambda T: np.arange(0, T, 1 / fs)
    W1, W2 = [], []
    u = blk(10); a = (25 * 2 * np.pi * .5) * (np.sin(2 * np.pi * .5 * u) - 1)
    W1.append(np.zeros((len(u), 3))); W2.append(a[:, None] * axis)
    u = blk(2); W1.append(np.zeros((len(u), 3))); W2.append(np.zeros((len(u), 3)))
    u = blk(10); b = (20 * 2 * np.pi * .5) * (np.sin(2 * np.pi * .5 * u) + 1)
    W1.append(b[:, None] * axis); W2.append(np.zeros((len(u), 3)))
    u = blk(2); W1.append(np.zeros((len(u), 3))); W2.append(np.zeros((len(u), 3)))
    u = blk(10); c = (35 * 2 * np.pi * .6) * np.sin(2 * np.pi * .6 * u)
    if good:
        W1.append(c[:, None] * axis); W2.append(c[:, None] * axis)
    else:
        # part C done wrong: the ankle was left loose instead of held stiff,
        # so the two sensors no longer see the same rotation
        wob = (8 * 2 * np.pi * 1.3) * np.sin(2 * np.pi * 1.3 * u)
        W1.append(c[:, None] * axis)
        W2.append((c + wob)[:, None] * axis)
    u = blk(2); W1.append(np.zeros((len(u), 3))); W2.append(np.zeros((len(u), 3)))
    w1, w2 = np.vstack(W1), np.vstack(W2)
    N = len(w1); t = np.arange(N) / fs + 1000.0
    g1 = R1.inv().apply(w1) + np.array([1.2, -.7, 2.1]) + r.normal(0, 1, (N, 3))
    g2 = R2.inv().apply(w2) + np.array([-3.4, .9, -1.1]) + r.normal(0, 1, (N, 3))
    acc = np.tile([0, 0, -1.0], (N, 1)) + r.normal(0, .02, (N, 3))
    mk = lambda g, nm: SensorStream(accel=acc, timestamps=t, actual_fs=fs,
                                    name=nm, gyro=g, side="left",
                                    backend="trigno")
    return mk(g2, "left_foot"), mk(g1, "left_shank")


def demo():
    print(f"\n{BOLD}What a good calibration looks like{OFF}")
    print(f"{DIM}  simulated, with a known ankle axis and 36° of motion{OFF}")
    foot, shank = _synth(good=True)
    report(functional_calibration(foot, shank), "left leg")

    print(f"\n{BOLD}What it looks like when part C is done wrong{OFF}")
    print(f"{DIM}  the ankle was left loose during the swing instead of held "
          f"stiff{OFF}")
    foot, shank = _synth(good=False)
    try:
        report(functional_calibration(foot, shank), "left leg")
    except ValueError as e:
        print(f"    {RED}refused:{OFF} {str(e).splitlines()[0]}")

    print(f"\n{BOLD}What a plain walking file looks like{OFF}")
    newest = _newest()
    if newest:
        check_file(newest)
    print()


def rehearse():
    print(f"\n{BOLD}Ankle axis calibration — rehearsal{OFF}\n")
    print("Mount the sensors where they will stay for the session, start a")
    print("recording, and do these three movements. Any order. Roughly ten")
    print("seconds each, and pause a beat between them.\n")
    for line in FUNCTIONAL_CALIBRATION.splitlines()[1:]:
        print("   " + line.strip())
    print(f"\n{DIM}Record them as their own file, before the walking trials.")
    print(f"Do not fold them into a walking trial: walking can be mistaken")
    print(f"for the third movement.{OFF}\n")
    print("Then stop the recording and run:\n")
    print(f"   {BOLD}./apps/check_calibration.py{OFF}\n")
    print("which checks the newest file. Do this before the participant takes")
    print("the sensors off — the axis encodes where each sensor sits, so a")
    print("failed calibration cannot be repaired afterwards.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="an .xdf recording")
    ap.add_argument("--demo", action="store_true",
                    help="show what pass and fail look like")
    ap.add_argument("--rehearse", action="store_true",
                    help="print the protocol and what to run afterwards")
    args = ap.parse_args()

    if args.rehearse:
        rehearse(); return 0
    if args.demo:
        demo(); return 0

    path = args.path or _newest()
    if path is None:
        print(f"No .xdf recordings found under {_root()}.")
        print("Run with --rehearse to see the protocol.")
        return 1
    if not args.path:
        print(f"{DIM}newest recording under {_root()}{OFF}")
    rc = check_file(path)
    if rc:
        print(f"\n{DIM}Run --rehearse for the protocol, or --demo to see what "
              f"a good calibration looks like.{OFF}")
    print()
    return rc


if __name__ == "__main__":
    sys.exit(main())
