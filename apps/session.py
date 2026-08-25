#!/usr/bin/env python3.12
"""
apps/session.py — one command to run a HITLO session.

    ./apps/session.py

Replaces the six-terminal dance with a single guided flow:

  0. clear any streamers left running from a previous session
  1. scan for Polar sensors; assign left/right if they changed
  2. pick the subject id, warning about existing data or a checkpoint
  3. start both sensor streams and confirm real samples are flowing
  4. verify which LEG each stream is on (the shake test)
  5. run preflight
  6. launch the console, and clean the streamers up on exit

Each step blocks until it passes, so you cannot get four steps in and discover
step one was wrong. Everything it does can still be done by hand with the
individual tools -- this only sequences them.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
PY = sys.executable

_cfg = REPO / 'config' / 'lsl_api.cfg'
if _cfg.is_file() and 'LSLAPICFG' not in os.environ:
    os.environ['LSLAPICFG'] = str(_cfg)

import yaml

CFG_PATH = REPO / 'config' / 'exo_symmetry_config.yml'
streamers = []


def hdr(n, title):
    print(f"\n{'='*66}\n  STEP {n} — {title}\n{'='*66}")


def die(msg):
    print(f"\n  STOPPED: {msg}")
    cleanup()
    sys.exit(1)


def ask(prompt, default=None):
    sfx = f" [{default}]" if default else ""
    try:
        v = input(f"  {prompt}{sfx}: ").strip()
    except EOFError:
        v = ""
    return v or (default or "")


def yes(prompt, default=True):
    d = "Y/n" if default else "y/N"
    v = ask(f"{prompt} ({d})").lower()
    return default if not v else v.startswith('y')


def cleanup(*_):
    for p in streamers:
        if p.poll() is None:
            p.terminate()
    if streamers:
        time.sleep(1)
        for p in streamers:
            if p.poll() is None:
                p.kill()


# --- 0 --------------------------------------------------------------------

def clear_stale():
    hdr(0, "CLEAR STALE PROCESSES")
    r = subprocess.run(['pgrep', '-f', 'collect_sensors.py'],
                       capture_output=True, text=True)
    pids = [p for p in r.stdout.split() if p.strip()]
    if not pids:
        print("  none running — clean start")
        return
    print(f"  found {len(pids)} existing streamer process(es): {' '.join(pids)}")
    print("  (duplicates fight over the same sensor and break both streams)")
    if yes("  kill them?"):
        for p in pids:
            subprocess.run(['kill', '-9', p], capture_output=True)
        time.sleep(2)
        print("  cleared")
    else:
        die("cannot start with duplicate streamers running")


# --- 1 --------------------------------------------------------------------

def sensors():
    hdr(1, "SENSORS")
    sf = REPO / 'config' / 'sensors.json'
    cur = json.loads(sf.read_text()) if sf.is_file() else {}
    print(f"  configured:  left={cur.get('left','?')}  right={cur.get('right','?')}")
    print("  scanning (15 s) ...")
    r = subprocess.run([PY, str(REPO/'apps'/'collect_sensors.py'),
                        '--scan', '--scan-timeout', '15'],
                       capture_output=True, text=True)
    print('\n'.join('  ' + l for l in r.stdout.splitlines() if l.strip()))
    found = set(re.findall(r'\b([0-9A-F]{8})\b', r.stdout))
    missing = [s for s in ('left', 'right') if cur.get(s) not in found]
    if missing:
        print(f"\n  NOT IN RANGE: {', '.join(cur.get(s,'?') for s in missing)}")
        print("  Wet the electrodes and put the strap on — the H10 stays asleep "
              "until\n  its two electrodes are bridged by skin.")
        if yes("  re-assign which strap is left/right?", default=True):
            subprocess.run([PY, str(REPO/'apps'/'collect_sensors.py'), '--assign'])
        else:
            die("both sensors must be in range")


# --- 2 --------------------------------------------------------------------

def subject():
    hdr(2, "SUBJECT")
    cfg = yaml.safe_load(CFG_PATH.read_text())
    cur = cfg['Subject']['id']
    print(f"  config currently says: {cur}")
    sid = ask("subject id", cur)
    ses = cfg['Subject']['session']
    base = Path(cfg['Subject']['base_dir'])
    from hitlo.io import backend_modality
    eeg = base / f"sub-{sid}" / f"ses-{ses}" / backend_modality(cfg)
    ck = (base / f"sub-{sid}" / f"ses-{ses}" / "derivatives" /
          "hil_optimization" / f"sub-{sid}_ses-{ses}_checkpoint.json")
    n = len(list(eeg.glob('*.xdf'))) if eeg.is_dir() else 0
    if n:
        print(f"\n  WARNING: {sid} already has {n} recording(s) in {eeg}")
    if ck.is_file():
        try:
            k = len(json.loads(ck.read_text()).get('results', []) or [])
        except Exception:
            k = '?'
        print(f"  WARNING: a checkpoint with {k} trial(s) exists — the console "
              f"will RESUME it,\n           not start fresh.")
        if not yes("  resume that session?", default=False):
            if yes(f"  delete the checkpoint and start {sid} over?", default=False):
                ck.unlink()
                print("  checkpoint deleted")
            else:
                die("pick a different subject id and re-run")
    elif n:
        if not yes(f"  continue with {sid} anyway?", default=False):
            die("pick a different subject id and re-run")
    if sid != cur:
        cfg['Subject']['id'] = sid
        yaml.safe_dump(cfg, CFG_PATH.open('w'), sort_keys=False,
                       default_flow_style=False)
        print(f"  config updated to {sid}")
    return sid


# --- 3 --------------------------------------------------------------------

def start_streams():
    hdr(3, "START SENSOR STREAMS")
    for side in ('right', 'left'):
        print(f"\n  starting {side} ...")
        p = subprocess.Popen(
            [PY, '-u', str(REPO/'apps'/'collect_sensors.py'), side,
             '--scan-timeout', '25'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}, cwd=str(REPO))
        streamers.append(p)
        ok = False
        t0 = time.time()
        while time.time() - t0 < 70:
            line = p.stdout.readline()
            if not line:
                if p.poll() is not None:
                    break
                continue
            s = line.rstrip()
            if any(k in s for k in ('Found', 'connected', 'samples in first',
                                    'REJECTED', 'No device found', 'ZERO samples')):
                print(f"    {s}")
            if 'samples in first' in s:
                ok = True
                break
            if 'No device found' in s or 'REJECTED' in s:
                break
        if not ok:
            die(f"{side} sensor did not start streaming")
    print("\n  both sensors streaming")


# --- 4, 5, 6 --------------------------------------------------------------

def verify_sides():
    hdr(4, "CONFIRM WHICH LEG IS WHICH")
    print("  A device id names a STRAP, not the leg it ended up on. A silent\n"
          "  left/right swap inverts the sign of the symmetry index.\n")
    if not yes("  run the shake test now?", default=True):
        print("  SKIPPED — sides unverified")
        return
    r = subprocess.run([PY, str(REPO/'apps'/'verify_sides.py')], cwd=str(REPO))
    if r.returncode != 0 and not yes("\n  sides NOT confirmed. continue anyway?",
                                     default=False):
        die("fix the strap placement and re-run")


def preflight():
    hdr(5, "PREFLIGHT")
    r = subprocess.run([PY, str(REPO/'apps'/'preflight.py')], cwd=str(REPO))
    if r.returncode != 0 and not yes("\n  preflight failed. continue anyway?",
                                     default=False):
        die("resolve the preflight failures first")


def console():
    hdr(6, "CONSOLE")
    print("  Launching. LabRecorder settings:")
    cfg = yaml.safe_load(CFG_PATH.read_text())
    sid, ses = cfg['Subject']['id'], cfg['Subject']['session']
    print(f"    check ONLY:  polar accel left   polar accel right")
    print(f"    save dir:    {Path(cfg['Subject']['base_dir'])/f'sub-{sid}'/f'ses-{ses}'/'eeg'}")
    print(f"    baseline:    task=Pre  run-001 (discarded), run-002 (the baseline)")
    print(f"    trials:      task=Default  run-001 .. run-{cfg['Optimization']['n_steps']:03d}")
    print("\n  Ctrl-C here stops the console AND both sensor streams.\n")
    subprocess.run([PY, '-m', 'streamlit', 'run',
                    str(REPO/'apps'/'hitlo_console.py')], cwd=str(REPO))


def main():
    signal.signal(signal.SIGINT, lambda *a: (cleanup(), sys.exit(130)))
    print("="*66); print("  HITLO SESSION"); print("="*66)
    try:
        clear_stale()
        sensors()
        subject()
        start_streams()
        verify_sides()
        preflight()
        console()
    finally:
        cleanup()
        print("\n  sensor streams stopped.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
