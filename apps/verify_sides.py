#!/usr/bin/env python3.12
"""
apps/verify_sides.py — prove which shank each LSL stream is actually on.

Run this AFTER both collect_sensors.py terminals are streaming, and BEFORE
you record anything.

    python apps/verify_sides.py

Why this exists:
  The device ID printed on a strap tells you which SENSOR you are holding.
  It tells you nothing about which LEG it ended up on. Straps get swapped
  during setup, and a swap is invisible in the data — you get a plausible
  symmetry index with the sign backwards. For Aim 1, where the whole point
  is driving SI toward a signed target, a silent L/R swap inverts the
  experiment.

  So: shake one leg at a time and watch which stream responds.

The test:
  1. Stand still  -> per-stream baseline motion
  2. Shake LEFT   -> 'polar accel left' must be the one that jumps
  3. Shake RIGHT  -> 'polar accel right' must be the one that jumps

Exit code 0 = sides confirmed, 1 = swapped or inconclusive.
"""

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Quiet liblsl before pylsl loads it.
_cfg = REPO_ROOT / 'config' / 'lsl_api.cfg'
if _cfg.is_file() and 'LSLAPICFG' not in os.environ:
    os.environ['LSLAPICFG'] = str(_cfg)

import numpy as np
from pylsl import StreamInlet, resolve_streams

def _sensing():
    """Backend from the live config; defaults to Polar.

    A missing config file is fine and means Polar. A config that exists but
    cannot be parsed is NOT fine and says so -- silently defaulting would run
    the whole check against the wrong backend, which is what a bare except
    here did until it swallowed a typo in this very function.
    """
    path = REPO_ROOT / 'config' / 'exo_symmetry_config.yml'
    cfg = {}
    if path.is_file():
        try:
            import yaml
            cfg = yaml.safe_load(path.read_text()) or {}
        except Exception as e:
            print(f"  WARNING: could not read {path.name} ({e}); "
                  f"assuming the Polar backend.")
    s = dict((cfg or {}).get('Sensing') or {})
    s.setdefault('backend', 'polar')
    s.setdefault('stream', 'TrignoIMU')
    return s


SENSING = _sensing()
WANT = ([SENSING['stream']] if SENSING['backend'] == 'trigno'
        else ['polar accel left', 'polar accel right'])
BASELINE_S = 3.0
SHAKE_S = 5.0
# The shaken stream must beat the OTHER stream by this ratio within the same
# window. This is the primary discriminator: comparing the two streams against
# each other is robust to a restless baseline, whereas ratio-to-own-baseline
# silently degrades if the subject moved while we measured "resting".
MIN_SEPARATION = 2.5
# Something must actually have happened: peak motion in the window has to
# exceed the quieter stream's resting level by this much.
MIN_ACTIVATION = 3.0
# Baselines this far apart mean the "stand still" window was not still.
BASELINE_IMBALANCE = 5.0


TRIGNO_COLS = None


def _trigno_cols(inlet):
    """Accel column indices per side, from an OPEN inlet's channel labels.

    LSL SUBTLETY: resolve_streams() returns metadata-LIGHT StreamInfo objects
    with an empty desc, so channel labels are NOT available there. The full
    description only arrives after opening an inlet and calling inlet.info().
    Reading labels off the resolved info silently yields none, which looks
    exactly like a bridge that forgot to declare them.
    """
    try:
        full = inlet.info(timeout=5.0)
        ch = full.desc().child('channels').child('channel')
        labels = []
        while not ch.empty():
            labels.append(ch.child_value('label').strip().lower())
            ch = ch.next_sibling()
    except Exception:
        return None
    out = {}
    for side in ('left', 'right'):
        cols = {}
        for i, lab in enumerate(labels):
            if not lab.startswith(side) or 'foot' in lab or 'thigh' in lab:
                continue
            if not any(t in lab for t in ('acc', 'accel')):
                continue
            for ax in ('x', 'y', 'z'):
                if lab.endswith(ax) and ax not in cols:
                    cols[ax] = i
        if len(cols) != 3:
            return None
        out[side] = [cols['x'], cols['y'], cols['z']]
    return out


def motion(inlet, seconds: float) -> float:
    """Std-dev of acceleration magnitude over a window. Resting ~ small."""
    t0 = time.time()
    mags = []
    inlet.flush()
    while time.time() - t0 < seconds:
        chunk, _ = inlet.pull_chunk(timeout=0.2, max_samples=256)
        for smp in chunk:
            if len(smp) >= 3:
                mags.append(float(np.sqrt(smp[0]**2 + smp[1]**2 + smp[2]**2)))
    if len(mags) < 10:
        return float('nan')
    return float(np.std(np.asarray(mags)))


def measure_both(inlets: dict, seconds: float) -> dict:
    """Both sides over the same wall-clock window.

    On Trigno the two sides share ONE inlet, so it is pulled once and split
    by column. Pulling per side would race, each call consuming samples the
    other never sees.
    """
    if TRIGNO_COLS is not None and inlets['left'] is inlets['right']:
        inl = inlets['left']
        inl.flush()
        mags = {'left': [], 'right': []}
        t0 = time.time()
        while time.time() - t0 < seconds:
            chunk, _ = inl.pull_chunk(timeout=0.2, max_samples=256)
            for smp in chunk:
                for sd in ('left', 'right'):
                    cx, cy, cz = TRIGNO_COLS[sd]
                    if len(smp) > max(cx, cy, cz):
                        mags[sd].append(
                            float(np.sqrt(smp[cx]**2 + smp[cy]**2 + smp[cz]**2)))
        return {sd: (float(np.std(np.asarray(v))) if len(v) >= 10 else float('nan'))
                for sd, v in mags.items()}

    import threading
    out = {}

    def work(side):
        out[side] = motion(inlets[side], seconds)

    ts = [threading.Thread(target=work, args=(s,)) for s in WANT_SIDES]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


WANT_SIDES = ['left', 'right']


def main() -> int:
    print("Resolving LSL streams (5s) ...")
    streams = resolve_streams(wait_time=5.0)
    if not streams:
        print("  No LSL streams at all. Are both collect_sensors.py running?")
        return 1

    import socket
    me = socket.gethostname().split('.')[0].lower()

    inlets, foreign = {}, []
    for s in streams:
        host = s.hostname().split('.')[0]
        # The own-host filter exists to stop another lab's identically-named
        # stream being used by mistake. It applies to Polar, whose bridge runs
        # locally. The Trigno bridge runs on the base-station machine BY
        # DESIGN, so for that backend the configured stream name is matched
        # from any host -- filtering by host would reject the real stream.
        remote_ok = (SENSING['backend'] == 'trigno' and
                     s.name() == SENSING['stream'])
        if host.lower() != me and not remote_ok:
            foreign.append((s.name(), host))
            continue
        if SENSING['backend'] == 'trigno':
            if s.name() == SENSING['stream']:
                inlet = StreamInlet(s, max_buflen=8)
                cols = _trigno_cols(inlet)
                if cols is None:
                    print(f"  '{s.name()}' declares no usable channel labels; "
                          f"cannot tell left from right.")
                    continue
                # ONE inlet feeds both sides; pulling it twice would race.
                inlets['left'] = inlets['right'] = inlet
                globals()['TRIGNO_COLS'] = cols
        elif s.name() == 'polar accel left':
            inlets['left'] = StreamInlet(s, max_buflen=8)
        elif s.name() == 'polar accel right':
            inlets['right'] = StreamInlet(s, max_buflen=8)

    if foreign:
        print("\n  Streams from OTHER machines (do not record these):")
        for name, host in foreign:
            note = "  <-- matches the single-sensor fallback name" \
                   if name == 'polar accel' else ""
            print(f"    - {name}  (host {host}){note}")

    missing = [s for s in WANT_SIDES if s not in inlets]
    if missing:
        print(f"\n  MISSING on this machine: {', '.join(missing)}")
        if SENSING['backend'] == 'trigno':
            print(f"  Expected the '{SENSING['stream']}' stream from the "
                  f"bridge on the base-station machine.")
        return 1
    print("  Both shank streams found on this machine.\n")

    input("STEP 1 — stand still, both sensors resting. Press ENTER ...")
    base = measure_both(inlets, BASELINE_S)
    for side in WANT_SIDES:
        if np.isnan(base[side]):
            print(f"  No data from the {side} stream. It is connected but "
                  f"not pushing samples.")
            return 1
    print(f"  baseline motion:  left={base['left']:.1f}  "
          f"right={base['right']:.1f}")
    lo, hi = min(base.values()), max(base.values())
    if hi > lo * BASELINE_IMBALANCE:
        print(f"  NOTE: baselines differ by {hi / (lo + 1e-6):.0f}x — someone "
              f"was moving during the still window.")
        print(f"  Sides are still decided by comparing the two streams to each "
              f"other, so this is survivable, but a quiet baseline is better.")
    print()

    results = {}
    for shaken in WANT_SIDES:
        input(f"STEP {2 if shaken == 'left' else 3} — shake ONLY the "
              f"{shaken.upper()} sensor for {SHAKE_S:.0f}s. Press ENTER ...")
        during = measure_both(inlets, SHAKE_S)
        other = 'right' if shaken == 'left' else 'left'
        quiet_ref = min(base.values()) + 1e-6
        print(f"  motion this window:  left={during['left']:.1f}  "
              f"right={during['right']:.1f}"
              f"   (resting ref {quiet_ref:.1f})")

        anything = max(during.values()) >= quiet_ref * MIN_ACTIVATION
        if not anything:
            verdict, ok = "NO MOTION DETECTED on either stream", False
        elif during[shaken] >= during[other] * MIN_SEPARATION:
            verdict, ok = f"correct — '{shaken}' stream responded", True
        elif during[other] >= during[shaken] * MIN_SEPARATION:
            verdict, ok = f"SWAPPED — the '{other}' stream responded", False
        else:
            verdict, ok = ("ambiguous — both streams moved together "
                           "(sensors on the same leg?)", False)
        print(f"  -> {verdict}\n")
        results[shaken] = ok

    print("=" * 60)
    if all(results.values()):
        print("SIDES CONFIRMED — 'polar accel left' is on the left shank.")
        print("Safe to record.")
        return 0

    print("SIDES NOT CONFIRMED. Do not record yet.")
    if not any(results.values()):
        print("\nIf both steps said SWAPPED, the straps are on the wrong legs.")
        print("Fix it in ONE of these ways:")
        if SENSING['backend'] == 'trigno':
            print("  On Trigno the side comes from the BRIDGE's slot-to-label")
            print("  mapping, not from anything on this machine. Either swap the")
            print("  sensors physically, or fix the mapping in the bridge on the")
            print("  base-station machine and restart it.")
        else:
            print("  a) physically swap the two straps, or")
            print("  b) swap left/right in config/sensors.json, then restart BOTH")
            print("     collect_sensors.py terminals (the stream names are set at")
            print("     connect time, so a config edit alone changes nothing).")
    print("\nThen re-run this check.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
