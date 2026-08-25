#!/usr/bin/env python3.12
"""
apps/preflight.py — run this before every session.

    python3.12 apps/preflight.py

Checks, in order:
  1. Regression tests (dtype handling, sign convention, ACC payload, ramp rows)
  2. Config sanity (subject id, aim, index table, target paradigm)
  3. Stale checkpoint that would silently resume a previous session
  4. LSL streams: present on THIS host, actually carrying samples,
     and no foreign stream that could be recorded by mistake
  5. End-to-end analysis on the most recent recording, if any

Exit 0 = clear to record. Exit 1 = something needs attention.

Written after a session where four separate faults all presented identically
as "nothing works": an int16 overflow that silently produced zero heel strikes,
an ACC configuration the sensor rejected without the script noticing, a 1s LSL
resolve window that falsely reported disconnected sensors, and a config still
pointing at a finished test subject.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_cfg = REPO / 'config' / 'lsl_api.cfg'
if _cfg.is_file() and 'LSLAPICFG' not in os.environ:
    os.environ['LSLAPICFG'] = str(_cfg)

import numpy as np
import yaml

from hitlo.io import backend_modality, live_stream_names

OK, WARN, BAD = "OK  ", "WARN", "FAIL"
issues = {'fail': 0, 'warn': 0}


def report(level, section, msg):
    if level == BAD:
        issues['fail'] += 1
    elif level == WARN:
        issues['warn'] += 1
    print(f"  [{level}] {section:22s} {msg}")


# --- 1. regression tests ---------------------------------------------------

def check_tests():
    print("\n1. REGRESSION TESTS")
    t = REPO / 'tests' / 'test_regression.py'
    if not t.is_file():
        report(BAD, "test suite", "tests/test_regression.py missing")
        return
    r = subprocess.run([sys.executable, str(t)], capture_output=True, text=True)
    last = [l for l in r.stdout.strip().splitlines() if 'passed' in l]
    summary = last[-1].strip() if last else "no summary"
    if r.returncode == 0:
        report(OK, "test suite", summary)
    else:
        report(BAD, "test suite", summary)
        for line in r.stdout.splitlines():
            if line.strip().startswith(('FAIL', 'ERROR')):
                print(f"         {line.strip()}")


# --- 2. config -------------------------------------------------------------

def check_config():
    print("\n2. CONFIG")
    p = REPO / 'config' / 'exo_symmetry_config.yml'
    if not p.is_file():
        report(BAD, "config file", "missing"); return None
    cfg = yaml.safe_load(p.read_text())
    sid = cfg['Subject']['id']
    ses = cfg['Subject']['session']
    report(OK, "subject", f"{sid} / {ses}   aim={cfg['Cost']['aim']}")

    base = Path(cfg['Subject']['base_dir'])
    # follows the backend: eeg/ for Polar, motion/ for Trigno
    eeg = base / f"sub-{sid}" / f"ses-{ses}" / backend_modality(cfg)
    existing = sorted(eeg.glob("*.xdf")) if eeg.is_dir() else []
    if existing:
        report(WARN, "existing data",
               f"{len(existing)} xdf already in {eeg.name}/ for {sid} — "
               f"confirm this is the right subject id")
    else:
        report(OK, "existing data", f"none for {sid} (clean session)")

    try:
        from hitlo.index_unified import IndexTable
        t = IndexTable(str(REPO / cfg['Optimization']['index_csv']))
        report(OK, "index table", f"{len(t)} levels, x in [-1, +1]")
    except Exception as e:
        report(BAD, "index table", f"{type(e).__name__}: {e}")

    if cfg['Cost'].get('signed') and cfg['Cost']['aim'].lower().startswith('aim 1'):
        report(OK, "paradigm", "Aim 1 signed — target comes from baseline "
                               "(Pre run-002), not si_target in this file")
    return cfg


# --- 3. checkpoint ---------------------------------------------------------

def check_checkpoint(cfg):
    print("\n3. CHECKPOINT")
    if cfg is None:
        return
    sid, ses = cfg['Subject']['id'], cfg['Subject']['session']
    p = (Path(cfg['Subject']['base_dir']) / f"sub-{sid}" / f"ses-{ses}" /
         "derivatives" / "hil_optimization" / f"sub-{sid}_ses-{ses}_checkpoint.json")
    if not p.is_file():
        report(OK, "checkpoint", "none — console will start fresh")
        return
    try:
        d = json.loads(p.read_text())
        n = len(d.get('results', []) or [])
        report(WARN, "checkpoint",
               f"EXISTS with {n} trial(s) for {sid} — the console will RESUME "
               f"this, not start over. Delete it or change subject id.")
    except Exception as e:
        report(WARN, "checkpoint", f"present but unreadable: {e}")


# --- 4. LSL ----------------------------------------------------------------

def check_lsl(cfg_for_streams=None):
    print("\n4. LSL STREAMS")
    import socket, time
    from pylsl import StreamInlet, resolve_streams
    me = socket.gethostname().split('.')[0].lower()
    streams = resolve_streams(wait_time=3.0)
    if not streams:
        report(BAD, "discovery", "no LSL streams at all — collect_sensors not running?")
        return

    mine, foreign = {}, []
    for s in streams:
        host = s.hostname().split('.')[0]
        if host.lower() == me:
            mine[s.name()] = s
        else:
            foreign.append((s.name(), host))

    for want in live_stream_names(cfg_for_streams):
        if want not in mine:
            report(BAD, want, "NOT PRESENT on this machine")
            continue
        inlet = StreamInlet(mine[want], max_buflen=4)
        inlet.flush()
        t0, n = time.time(), 0
        while time.time() - t0 < 3.0:
            chunk, _ = inlet.pull_chunk(timeout=0.2, max_samples=256)
            n += len(chunk)
        hz = n / 3.0
        if n == 0:
            report(BAD, want, "present but ZERO samples — sensor not sending")
        elif hz < 150:
            report(WARN, want, f"only ~{hz:.0f} Hz (expect ~200) — dropping samples")
        else:
            report(OK, want, f"~{hz:.0f} Hz")

    for name, host in foreign:
        lvl = BAD if name == 'polar accel' else WARN
        note = ("matches the single-sensor fallback name — if recorded, you get a "
                "symmetry index from someone else's data"
                if name == 'polar accel' else "another experiment; do not record")
        report(lvl, f"foreign: {name}", f"host {host} — {note}")


# --- 5. most recent recording ---------------------------------------------

def check_last_recording(cfg):
    print("\n5. LAST RECORDING")
    if cfg is None:
        return
    sid, ses = cfg['Subject']['id'], cfg['Subject']['session']
    eeg = (Path(cfg['Subject']['base_dir']) / f"sub-{sid}" / f"ses-{ses}"
           / backend_modality(cfg))
    files = sorted(eeg.glob("*.xdf"), key=lambda f: f.stat().st_mtime) if eeg.is_dir() else []
    if not files:
        report(OK, "analysis", "no recordings yet — nothing to verify")
        return
    latest = files[-1]
    from hitlo.cost import SymmetryCost
    import io as _io, contextlib
    buf = _io.StringIO()
    # SymmetryCost prints a banner from its constructor, so build it inside the
    # redirect too, not just the analysis call.
    with contextlib.redirect_stdout(buf):
        # Pass the config, or the cost function defaults to the Polar loader
        # and reports "no usable accel stream" on a perfectly good Trigno file.
        c = SymmetryCost(trial_data_dir=str(eeg), subject_id=sid, session=ses,
                         signed=cfg['Cost'].get('signed', True),
                         si_target=float(cfg['Cost'].get('si_target', 0.0)),
                         trim_seconds=float(cfg['Cost'].get('trim_seconds', 3.0)),
                         config=cfg)
        a = c.analyze_trial(trial_num=1, filename=latest.name, verbose=False)
    if a is None:
        report(BAD, latest.name[-28:], f"analysis failed: {c.last_failure}")
    else:
        report(OK, latest.name[-28:],
               f"L={len(a.left_heel_strikes)} R={len(a.right_heel_strikes)} "
               f"SI={a.symmetry_index:+.2f}%")
        for w in a.warnings:
            report(WARN, "  qc", w)


def main() -> int:
    print("=" * 70)
    print("HITLO PREFLIGHT")
    print("=" * 70)
    check_tests()
    cfg = check_config()
    check_checkpoint(cfg)
    try:
        check_lsl(cfg)
    except Exception as e:
        report(BAD, "LSL", f"{type(e).__name__}: {e}")
    try:
        check_last_recording(cfg)
    except Exception as e:
        report(BAD, "analysis", f"{type(e).__name__}: {e}")

    print("\n" + "=" * 70)
    if issues['fail']:
        print(f"NOT READY — {issues['fail']} failure(s), {issues['warn']} warning(s)")
    elif issues['warn']:
        print(f"READY WITH WARNINGS — {issues['warn']} item(s) to confirm")
    else:
        print("READY — all checks passed")
    print("=" * 70)
    return 1 if issues['fail'] else 0


if __name__ == '__main__':
    sys.exit(main())
