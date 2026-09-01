#!/usr/bin/env python3.12
"""
apps/make_fake_trial.py — write a synthetic XDF the console will accept as a trial.

    ./apps/make_fake_trial.py --subject P099 --run 1 --si -4
    ./apps/make_fake_trial.py --subject P099 --baseline      # Pre run-001 and run-002
    ./apps/make_fake_trial.py --subject P099 --session-set   # a whole ramp of trials

Then point the console at that subject and click through it for real: the file
is discovered, previewed, analysed and accepted by the same code a Trigno
recording goes through.

WHY THIS EXISTS
---------------
apps/dry_run.py substitutes a fake cost extractor, so the optimizer is
exercised but nothing ever reads a file. That gap hid a real bug: the console
built SymmetryCost without passing the config, so the optimizer's extractor ran
the Polar accelerometer detector and failed on every Trigno file. Recording,
viewing and reading a symmetry index all worked, because each uses a different
object -- the failure only appeared at the moment a trial was accepted.

This writes real XDF bytes so that path cannot go untested again.

WHAT IS SIMULATED
-----------------
Two shank IMUs walking with a requested step-time symmetry index. The gait is a
smooth template, not measured data: it produces clean detectable heel strikes
with a known asymmetry, which is what testing the console needs. It is NOT a
model of anyone's walking and nothing about a real subject should be inferred
from it.
"""

import argparse
import math
import os
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

SAMPLE_RATE = 148.148


# ---------------------------------------------------------------------------
# Minimal XDF writer
# ---------------------------------------------------------------------------
#
# XDF is a sequence of length-prefixed chunks after a 4-byte magic. Only the
# four chunk types a reader needs are written here: file header, stream header,
# samples, stream footer. Format spec: github.com/sccn/xdf.

TAG_FILEHEADER, TAG_STREAMHEADER, TAG_SAMPLES, TAG_STREAMFOOTER = 1, 2, 3, 6


def _varlen(n):
    """XDF's variable-length integer: a byte count, then the value."""
    if n < 256:
        return bytes([1]) + struct.pack("<B", n)
    if n < 2 ** 32:
        return bytes([4]) + struct.pack("<I", n)
    return bytes([8]) + struct.pack("<Q", n)


def _chunk(tag, content):
    payload = struct.pack("<H", tag) + content
    return _varlen(len(payload)) + payload


def _stream_header_xml(name, stype, labels, srate, source_id):
    chans = "".join(
        f"<channel><label>{lab}</label>"
        f"<unit>{'g' if '_acc' in lab else 'deg/s'}</unit>"
        f"<type>{'ACC' if '_acc' in lab else 'GYR'}</type></channel>"
        for lab in labels)
    return (
        '<?xml version="1.0"?>'
        "<info>"
        f"<name>{name}</name>"
        f"<type>{stype}</type>"
        f"<channel_count>{len(labels)}</channel_count>"
        f"<nominal_srate>{srate}</nominal_srate>"
        "<channel_format>float32</channel_format>"
        f"<source_id>{source_id}</source_id>"
        "<version>1.100000</version>"
        f"<created_at>0.0</created_at>"
        "<uid>fake-trial</uid>"
        "<session_id>default</session_id>"
        "<hostname>synthetic</hostname>"
        f"<desc><channels>{chans}</channels>"
        "<acquisition><manufacturer>synthetic</manufacturer></acquisition>"
        "</desc>"
        "</info>").encode()


def write_xdf(path, name, labels, data, timestamps, stype="IMU"):
    """One float32 stream, timestamps on every sample."""
    data = np.asarray(data, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    n_samp, n_ch = data.shape
    sid = 1

    out = bytearray(b"XDF:")
    out += _chunk(TAG_FILEHEADER,
                  b'<?xml version="1.0"?><info><version>1.0</version></info>')
    out += _chunk(TAG_STREAMHEADER,
                  struct.pack("<I", sid)
                  + _stream_header_xml(name, stype, labels, SAMPLE_RATE, "synthetic"))

    # Sample chunks, blocked so no single chunk is enormous.
    BLOCK = 2048
    for start in range(0, n_samp, BLOCK):
        stop = min(start + BLOCK, n_samp)
        body = bytearray(struct.pack("<I", sid))
        body += _varlen(stop - start)
        for i in range(start, stop):
            body += bytes([8]) + struct.pack("<d", float(timestamps[i]))
            body += data[i].tobytes()
        out += _chunk(TAG_SAMPLES, bytes(body))

    footer = (
        '<?xml version="1.0"?><info><first_timestamp>'
        f"{timestamps[0]}</first_timestamp><last_timestamp>"
        f"{timestamps[-1]}</last_timestamp><sample_count>{n_samp}"
        "</sample_count></info>").encode()
    out += _chunk(TAG_STREAMFOOTER, struct.pack("<I", sid) + footer)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(bytes(out))
    return path


# ---------------------------------------------------------------------------
# Synthetic gait
# ---------------------------------------------------------------------------

def _shank_gyro_template(phase):
    """Sagittal shank angular velocity over one stride, heel strike at phase 0.

    Shaped to carry the two landmarks the gyro detector uses: a large mid-swing
    peak, then a negative-going zero crossing at the next heel strike. Values
    are in deg/s and in the right ballpark for walking, but this is a drawn
    curve, not measured data.
    """
    p = np.asarray(phase)
    swing = 260.0 * np.exp(-(((p - 0.78) / 0.085) ** 2))     # mid-swing peak
    pre_hs = -150.0 * np.exp(-(((p - 0.97) / 0.030) ** 2))   # dive into contact
    loading = -55.0 * np.exp(-(((p - 0.06) / 0.045) ** 2))   # foot-flat dip
    stance = 22.0 * np.sin(np.pi * np.clip(p / 0.6, 0, 1))   # slow stance roll
    return swing + pre_hs + loading + stance


def make_trial(si_percent=-4.0, seconds=90.0, stride_s=1.40, walk_from=6.0,
               dominance=2.6, noise=3.0, seed=0):
    """Two shank IMUs walking with the requested step-time symmetry index.

    SI = 2 (right_step - left_step) / (right_step + left_step) x 100, where a
    right step is the gap from a left heel strike to the next right one. With
    both legs on the same stride time T and the right foot landing d after the
    left, right_step = d and left_step = T - d, so

        SI = 200 (2d - T) / T      ->      d = T/2 x (1 + SI/200)

    The 200 is not a typo and the factor of two is easy to lose: writing /100
    here produces exactly double the requested asymmetry, which looks plausible
    and is only caught by measuring the file back.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    d = 0.5 * stride_s * (1.0 + si_percent / 200.0)

    # Standing before and after, so walking_window() has something to find and
    # the trim has something to trim.
    walk_to = seconds - 4.0
    walking = (t >= walk_from) & (t <= walk_to)

    def leg(offset):
        ph = np.zeros(n)
        ph[walking] = ((t[walking] - walk_from - offset) / stride_s) % 1.0
        w = np.where(walking, _shank_gyro_template(ph), 0.0)
        w = w + rng.normal(0, noise, n)
        return w

    w_left = leg(0.0)
    w_right = leg(d)

    # Distribute onto three axes. The sagittal axis gets `dominance` times the
    # standard deviation of the next, which is what the console's mounting
    # check reads -- so a low value here reproduces a loose mounting.
    def axes(w):
        secondary = rng.normal(0, np.std(w) / max(dominance, 1e-6), n)
        tertiary = rng.normal(0, np.std(w) / (max(dominance, 1e-6) * 3), n)
        return np.column_stack([tertiary, secondary, w])       # sagittal on z

    def accel(w):
        g = np.column_stack([np.full(n, 0.05), np.full(n, 0.10),
                             np.full(n, 0.993)])
        shock = np.zeros(n)
        shock[walking] = 0.35 * np.sin(2 * np.pi * t[walking] / stride_s * 2)
        return g + np.column_stack([shock, shock * 0.5, shock * 0.8]) \
            + rng.normal(0, 0.02, (n, 3))

    data = np.hstack([accel(w_left), axes(w_left),
                      accel(w_right), axes(w_right)])
    labels = ([f"left_shank_acc_{a}" for a in "xyz"]
              + [f"left_shank_gyr_{a}" for a in "xyz"]
              + [f"right_shank_acc_{a}" for a in "xyz"]
              + [f"right_shank_gyr_{a}" for a in "xyz"])
    # LSL timestamps are seconds since an arbitrary epoch, never zero-based.
    return data, labels, t + 100000.0


# ---------------------------------------------------------------------------

def _path_for(base_dir, subject, session, run, task, modality="motion"):
    fname = (f"sub-{subject}_ses-{session}_task-{task}_run-{run:03d}"
             f"_{modality}.xdf")
    return os.path.join(base_dir, f"sub-{subject}", f"ses-{session}",
                        modality, fname)


def emit(base_dir, subject, session, run, task, si, args, seed):
    data, labels, ts = make_trial(si_percent=si, seconds=args.seconds,
                                  dominance=args.dominance, noise=args.noise,
                                  seed=seed)
    p = _path_for(base_dir, subject, session, run, task, args.modality)
    write_xdf(p, "TrignoIMU", labels, data, ts)
    print(f"  wrote {os.path.basename(p)}   requested SI {si:+.1f}%   "
          f"{len(ts) / SAMPLE_RATE:.0f}s")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="P099")
    ap.add_argument("--session", default="S001")
    ap.add_argument("--base-dir", default=os.path.expanduser("~/HITLO_Data"))
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--task", default="Default")
    ap.add_argument("--modality", default="motion",
                    help="'motion' for Trigno, 'eeg' for the Polar template")
    ap.add_argument("--si", type=float, default=-4.0,
                    help="step-time symmetry index to synthesise, in percent")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--dominance", type=float, default=2.6,
                    help="sagittal axis dominance; below 1.6 trips the "
                         "console's mounting warning")
    ap.add_argument("--noise", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", action="store_true",
                    help="write Pre run-001 and run-002 instead of one trial")
    ap.add_argument("--session-set", type=int, metavar="N",
                    help="write the baseline plus N Default trials")
    args = ap.parse_args()

    print(f"\nsub-{args.subject} / ses-{args.session} -> "
          f"{args.base_dir}/sub-{args.subject}/ses-{args.session}/{args.modality}/\n")

    if args.baseline or args.session_set:
        emit(args.base_dir, args.subject, args.session, 1, "Pre", args.si, args, 1)
        emit(args.base_dir, args.subject, args.session, 2, "Pre", args.si, args, 2)
    if args.session_set:
        # Trials drift toward the target so the console shows movement rather
        # than a flat line. No claim that a device behaves this way.
        target = args.si - 10.0
        for k in range(1, args.session_set + 1):
            frac = k / max(args.session_set, 1)
            si = args.si + (target - args.si) * frac
            emit(args.base_dir, args.subject, args.session, k, "Default",
                 si, args, 100 + k)
    elif not args.baseline:
        emit(args.base_dir, args.subject, args.session, args.run, args.task,
             args.si, args, args.seed)

    print(f"\nPoint the console at subject {args.subject}, session "
          f"{args.session} and step through it.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
