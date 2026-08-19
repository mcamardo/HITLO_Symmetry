#!/usr/bin/env python3.12
"""
apps/check_placement.py — compare sensor mounting positions empirically.

    ./apps/check_placement.py            # 30 s walk, then a verdict
    ./apps/check_placement.py --seconds 20 --label "coban both legs"

Walk normally for the duration, then it reports, per leg:

  margin   how far the accepted heel strike beats the peaks it rejected,
           in SD of the jerk signal. This is the number that matters --
           sub-P997 had RIGHT 4.41 and LEFT 0.16, meaning the left detector
           was choosing almost at random among comparable peaks.
  amp      magnitude std, i.e. how much impact energy reaches the sensor
  ratio    the two legs against each other

Run it once per mounting you want to try and compare. Requires both streams
to be live (start them from the console's Sensors page).
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
_cfg = REPO / 'config' / 'lsl_api.cfg'
if _cfg.is_file() and 'LSLAPICFG' not in os.environ:
    os.environ['LSLAPICFG'] = str(_cfg)

import numpy as np
from pylsl import StreamInlet, resolve_streams

from hitlo.detection import (DetectionConfig, detect_heelstrikes_full,
                             compute_jerk_z, compute_magnitude)

GOOD, POOR = 2.0, 0.8


def collect(seconds):
    import socket
    me = socket.gethostname().split('.')[0].lower()
    inlets = {}
    for s in resolve_streams(wait_time=3.0):
        if s.hostname().split('.')[0].lower() != me:
            continue
        if s.name() == 'polar accel left':
            inlets['left'] = StreamInlet(s, max_buflen=120)
        elif s.name() == 'polar accel right':
            inlets['right'] = StreamInlet(s, max_buflen=120)
    missing = [s for s in ('left', 'right') if s not in inlets]
    if missing:
        print(f"  missing stream(s): {', '.join(missing)} — start them first")
        return None
    for i in inlets.values():
        i.flush()
    print(f"  walk normally for {seconds:.0f} s ...")
    buf = {k: {'a': [], 't': []} for k in inlets}
    t0 = time.time()
    while time.time() - t0 < seconds:
        for k, inl in inlets.items():
            ch, ts = inl.pull_chunk(timeout=0.05, max_samples=512)
            if ch:
                buf[k]['a'].extend(ch)
                buf[k]['t'].extend(ts)
        left = seconds - (time.time() - t0)
        print(f"\r  {left:5.1f} s remaining", end='', flush=True)
    print()
    return buf


def score(a, t, cfg):
    a = np.asarray(a, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(a) < cfg.fs * 5:
        return None
    res = detect_heelstrikes_full(a, t, cfg=cfg)
    jz, mag = compute_jerk_z(a, cfg)
    acc = res.heel_strike_indices
    rej = res.rejected_peaks
    if len(acc) < 5 or len(rej) < 1:
        return None
    return dict(n=len(acc),
                margin=float(np.mean(jz[acc]) - np.mean(jz[np.asarray(rej, int)])),
                amp=float(np.std(mag)),
                rate=len(acc) / (t[-1] - t[0]) * 60)


def main():
    ap = argparse.ArgumentParser(description="Compare sensor mounting positions.")
    ap.add_argument('--seconds', type=float, default=30.0)
    ap.add_argument('--label', default='')
    a = ap.parse_args()

    print("=" * 60)
    print(f"  PLACEMENT CHECK{'  —  ' + a.label if a.label else ''}")
    print("=" * 60)
    buf = collect(a.seconds)
    if buf is None:
        return 1
    cfg = DetectionConfig()
    out = {}
    for side in ('left', 'right'):
        s = score(buf[side]['a'], buf[side]['t'], cfg)
        if s is None:
            print(f"  {side}: not enough clean data")
            return 1
        out[side] = s
    print(f"\n  {'':6s} {'strikes':>8} {'/min':>7} {'amp':>8} {'margin':>8}")
    for side in ('left', 'right'):
        s = out[side]
        flag = "  GOOD" if s['margin'] >= GOOD else (
               "  POOR — detector is guessing" if s['margin'] < POOR else "  marginal")
        print(f"  {side:6s} {s['n']:8d} {s['rate']:7.0f} {s['amp']:8.0f} "
              f"{s['margin']:8.2f}{flag}")
    lm, rm = out['left']['margin'], out['right']['margin']
    la, ra = out['left']['amp'], out['right']['amp']
    print(f"\n  amplitude  left/right = {la/ra:.2f}   (want > 0.7)")
    print(f"  margin     left/right = {lm/max(rm,1e-9):.2f}   (want > 0.5)")
    print()
    if lm >= GOOD and rm >= GOOD:
        print("  Both legs detect decisively. This mounting is good.")
    elif lm < POOR:
        print("  LEFT is still guessing. Try, in order: Coban on the left too;")
        print("  move onto the flat bony shin above the ankle; move proximal to")
        print("  the exo cuff so impact travels through bone, not the device.")
    else:
        print("  Left is usable but weaker than right — worth one more position.")
    print("\n  Re-run after each change and keep the mounting with the highest")
    print("  left margin.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
