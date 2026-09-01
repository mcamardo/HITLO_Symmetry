#!/usr/bin/env python3
"""
trigno_lsl_bridge.py - republish Trigno Avanti IMU channels as an LSL stream.

Queries the base station over TCP, works out each sensor's channel block from
its STARTINDEX, and republishes the accelerometer and gyroscope channels as an
LSL stream named "TrignoIMU" at 148.148 Hz, for LabRecorder on another machine
to record.

Sensor modes must match across all paired sensors - mode 609 on our units, IMU
only, no EMG channel. One sensor in a different mode is allocated a different
block size and shifts every subsequent sensor's channels, which relabels body
parts without any error appearing. See docs/trigno_setup.md.

Startup order matters and is not arbitrary:

    discover -> START -> identify sides -> measure gyro bias -> open outlet

Both the side identification and the bias measurement need frames flowing, so
they happen after START. The outlet is opened last because its channel labels
come from the identification step and its metadata carries the measured bias.

Python 3.10, Windows. No dependencies beyond pylsl and the standard library.
"""

import argparse
import socket
import struct
import sys
import time

from pylsl import StreamInfo, StreamOutlet, local_clock

# ---------------------------------------------------------------------------
# Base station
# ---------------------------------------------------------------------------

HOST = "localhost"
CMD_PORT = 50040          # command / control
AUX_PORT = 50044          # AUX data: accelerometer + gyroscope. NOT EMG.

MAX_SENSORS = 16
AUX_STRIDE = 9            # channels allocated per sensor block
AUX_TOTAL = MAX_SENSORS * AUX_STRIDE
FRAME_BYTES = AUX_TOTAL * 4       # float32 little-endian

CHANS_PER_SENSOR = 6      # acc x/y/z then gyr x/y/z, the first six of the nine
ACC_OFFSETS = (0, 1, 2)
GYR_OFFSETS = (3, 4, 5)

SAMPLE_RATE = 148.148

# Slot -> body position. Used only as a fallback now: the interactive
# identification below assigns sides from what actually moves, because slot
# numbers shift whenever sensors are paired or re-moded.
SENSOR_MAP = {
    1: "left_shank",
    3: "right_shank",
}

# Accelerometer sanity band, in g, for a sensor sitting still.
ACC_MIN_G = 0.85
ACC_MAX_G = 1.15


# ---------------------------------------------------------------------------
# Command channel
# ---------------------------------------------------------------------------

def command(sock, text):
    """Send one command and return the base station's reply, stripped."""
    sock.sendall((text + "\r\n\r\n").encode())
    return sock.recv(1024).decode(errors="replace").strip()


def discover_sensors(cmd):
    """Every paired sensor, with the channel block the base station gave it.

    STARTINDEX is queried rather than derived from the slot number: it counts
    PAIRED sensors, not slots, so slot 3 reports 2 as soon as slot 2 is empty.
    Computing the base from the slot is the single most common way to end up
    reading one sensor's data under another's label.
    """
    found = []
    for slot in range(1, MAX_SENSORS + 1):
        if command(cmd, f"SENSOR {slot} PAIRED?").upper() != "YES":
            continue
        start = command(cmd, f"SENSOR {slot} STARTINDEX?")
        mode = command(cmd, f"SENSOR {slot} MODE?")
        serial = command(cmd, f"SENSOR {slot} SERIAL?")
        try:
            start_i = int(start)
        except ValueError:
            print(f"  slot {slot}: STARTINDEX came back as {start!r}, skipping")
            continue
        base = (start_i - 1) * AUX_STRIDE
        found.append(dict(slot=slot, startindex=start_i, base=base,
                          mode=mode, serial=serial))
    return found


def warn_about_modes(sensors):
    """Mixed modes shift channel offsets. Say so loudly, before any data."""
    modes = {s["mode"] for s in sensors}
    if len(modes) > 1:
        print("\n  ** WARNING: sensors are in different modes: "
              + ", ".join(sorted(modes)))
        print("     A mode with an EMG channel is allocated differently and")
        print("     shifts every later sensor's channels. Body parts will be")
        print("     mislabelled with no error. Fix in the Trigno Control")
        print("     Utility before recording.\n")


# ---------------------------------------------------------------------------
# Data channel
# ---------------------------------------------------------------------------

class FrameReader:
    """Whole AUX frames off the data socket.

    recv() returns whatever happens to have arrived, which is rarely a frame
    boundary, so buffer and hand back only complete frames.
    """

    def __init__(self, sock):
        self.sock = sock
        self.buf = bytearray()

    def next_frame(self):
        while len(self.buf) < FRAME_BYTES:
            chunk = self.sock.recv(8192)
            if not chunk:
                return None
            self.buf.extend(chunk)
        raw = bytes(self.buf[:FRAME_BYTES])
        del self.buf[:FRAME_BYTES]
        return struct.unpack("<%df" % AUX_TOTAL, raw)


def sensor_slice(frame, base):
    """The six channels this sensor owns: acc x/y/z then gyr x/y/z."""
    return frame[base:base + CHANS_PER_SENSOR]


def gyro_magnitude(frame, base):
    gx, gy, gz = (frame[base + o] for o in GYR_OFFSETS)
    return (gx * gx + gy * gy + gz * gz) ** 0.5


def accel_magnitude(frame, base):
    ax, ay, az = (frame[base + o] for o in ACC_OFFSETS)
    return (ax * ax + ay * ay + az * az) ** 0.5


def is_blank_frame(frame, sensors):
    """A frame with nothing in it yet.

    The base station emits zero-filled frames immediately after START, before
    the sensors' data actually arrives. They are not measurements and must not
    be averaged into anything: a single one drags a ten-frame accelerometer
    mean from 1.00 to 0.90 g, and pulls a bias estimate toward zero.

    Every sensor reading exactly zero means the stream has not started. One
    sensor reading zero while others do not is a real fault, so that frame is
    kept and allowed to trip the checks below.
    """
    return all(accel_magnitude(frame, s["base"]) < 1e-9 for s in sensors)


# ---------------------------------------------------------------------------
# 1. Accelerometer sanity check, averaged
# ---------------------------------------------------------------------------

class AccelSanityCheck:
    """Is every sensor reading about 1 g while it sits still?

    Averaged over roughly a second rather than taken from one frame. The first
    AUX frame after START is zero-filled, so a single-frame check reported
    "accel magnitude 0.00 g" on every launch and then latched, which trained
    everyone to ignore it. A second of frames also rides out the occasional
    dropped or partial frame.
    """

    def __init__(self, sensors, n_frames=int(SAMPLE_RATE)):
        self.sensors = sensors
        self.n_frames = max(int(n_frames), 1)
        self.sums = {s["slot"]: 0.0 for s in sensors}
        self.count = 0
        self.checked = False

    def feed(self, frame):
        """Accumulate one frame; report once enough have arrived."""
        if self.checked or is_blank_frame(frame, self.sensors):
            return
        for s in self.sensors:
            self.sums[s["slot"]] += accel_magnitude(frame, s["base"])
        self.count += 1
        if self.count < self.n_frames:
            return

        self.checked = True
        bad = []
        print(f"\n  accelerometer check, mean over {self.count} frames:")
        for s in self.sensors:
            mean_g = self.sums[s["slot"]] / self.count
            ok = ACC_MIN_G <= mean_g <= ACC_MAX_G
            print(f"    slot {s['slot']:>2}  {s['label']:<12} {mean_g:5.2f} g"
                  + ("" if ok else "   <-- out of range"))
            if not ok:
                bad.append((s, mean_g))
        if bad:
            print(f"\n  ** WARNING: {len(bad)} sensor(s) outside "
                  f"{ACC_MIN_G:.2f}-{ACC_MAX_G:.2f} g while stationary.")
            print("     Usually the sensor was moving during the check. If it")
            print("     was still, suspect the channel mapping or a sensor")
            print("     that has stopped reporting.\n")


# ---------------------------------------------------------------------------
# 2. Quiet-standing gyro bias
# ---------------------------------------------------------------------------

def measure_gyro_bias(reader, sensors, seconds=5.0):
    """Per-axis gyroscope offset for each sensor, measured standing still.

    These offsets are real and large enough to matter: -8.8, -6.1 and -0.8
    deg/s on one of our sensors. Anything that integrates the gyro accumulates
    them without bound - 8.8 deg/s is 8.8 degrees of error per second - so they
    are subtracted at the source and recorded in the stream metadata, which
    means the XDF says what was taken off rather than leaving it to be guessed
    later.

    Accelerometer channels are deliberately left alone: their offset is not
    separable from gravity without knowing the sensor's orientation.

    Returns {slot: [bias_gx, bias_gy, bias_gz]}.
    """
    n = max(int(seconds * SAMPLE_RATE), 1)
    print(f"\n  Gyro bias: hold still for {seconds:.0f} seconds.")
    input("  Stand still, then press Enter to start measuring... ")

    sums = {s["slot"]: [0.0, 0.0, 0.0] for s in sensors}
    peak = {s["slot"]: 0.0 for s in sensors}
    got = 0
    while got < n:
        frame = reader.next_frame()
        if frame is None:
            print("  data socket closed during bias measurement")
            break
        if is_blank_frame(frame, sensors):
            continue
        for s in sensors:
            for i, off in enumerate(GYR_OFFSETS):
                sums[s["slot"]][i] += frame[s["base"] + off]
            peak[s["slot"]] = max(peak[s["slot"]],
                                  gyro_magnitude(frame, s["base"]))
        got += 1

    bias = {}
    print(f"\n  measured over {got} frames:")
    for s in sensors:
        b = [v / got for v in sums[s["slot"]]] if got else [0.0, 0.0, 0.0]
        bias[s["slot"]] = b
        note = ""
        # A stationary sensor peaks a few deg/s above its own offset. Much more
        # than that and the subject was moving, which biases the bias.
        if peak[s["slot"]] > 25.0:
            note = f"   <-- moved during measurement (peak {peak[s['slot']]:.0f} deg/s)"
        print(f"    slot {s['slot']:>2}  {s['label']:<12} "
              f"[{b[0]:+6.2f} {b[1]:+6.2f} {b[2]:+6.2f}] deg/s{note}")
    return bias


# ---------------------------------------------------------------------------
# 3. Interactive left/right identification
# ---------------------------------------------------------------------------

def identify_sides(reader, sensors, labels_wanted, seconds=3.0,
                   margin=3.0, move_dps=40.0):
    """Assign body positions by shaking one sensor at a time.

    Slot numbers move whenever sensors are paired, unpaired or re-moded, so a
    hardcoded slot->body map goes stale silently and inverts the symmetry index
    while producing entirely plausible numbers. Asking which sensor is moving
    takes fifteen seconds and cannot go stale.

    Requires the shaken sensor to clearly dominate: if two sensors are within
    `margin` of each other the shake was too gentle or the wrong sensor moved,
    and the prompt repeats rather than guessing.

    Returns {slot: label}, or None if the user skips.
    """
    print("\n  Side identification: shake one sensor at a time.")
    print("  Enter 's' at any prompt to skip and fall back to SENSOR_MAP.")

    pool = list(sensors)
    assigned = {}

    for label in labels_wanted:
        while True:
            if len(pool) == 1:
                s = pool[0]
                print(f"    {label}: only slot {s['slot']} left, assigning it")
                assigned[s["slot"]] = label
                pool.remove(s)
                break

            # Enter means "ready", 's' means skip. Deliberately not
            # overloading Enter to mean both: the operator is holding a
            # participant's leg and should not have to remember which prompt
            # they are on.
            ans = input(f"    Ready to shake the {label.upper()} sensor? "
                        f"Enter to start, 's' to skip all: ").strip()
            if ans.lower() == "s":
                return None

            n = max(int(seconds * SAMPLE_RATE), 1)
            sums = {s["slot"]: 0.0 for s in pool}
            print(f"      shake now, {seconds:.0f} s...", end="", flush=True)
            got = 0
            while got < n:
                frame = reader.next_frame()
                if frame is None:
                    print(" data socket closed")
                    return None
                if is_blank_frame(frame, sensors):
                    continue
                for s in pool:
                    sums[s["slot"]] += gyro_magnitude(frame, s["base"])
                got += 1
            print(" done")

            ranked = sorted(((sums[s["slot"]] / got, s) for s in pool),
                            key=lambda p: p[0], reverse=True)
            top_mag, top = ranked[0]
            runner = ranked[1][0] if len(ranked) > 1 else 0.0

            for mag, s in ranked:
                print(f"        slot {s['slot']:>2}  {mag:7.1f} deg/s mean")

            if top_mag < move_dps:
                print(f"      nothing moved enough (best {top_mag:.0f} deg/s, "
                      f"need {move_dps:.0f}). Shake harder and try again.")
                continue
            if runner > 0 and top_mag < margin * runner:
                print(f"      two sensors moved similarly "
                      f"({top_mag:.0f} vs {runner:.0f} deg/s). Shake only the "
                      f"{label} sensor and try again.")
                continue

            print(f"      -> slot {top['slot']} is {label}")
            assigned[top["slot"]] = label
            pool.remove(top)
            break

    return assigned


# ---------------------------------------------------------------------------
# Outlet
# ---------------------------------------------------------------------------

def build_outlet(sensors, bias, source_id):
    """LSL outlet with per-channel labels and the bias that was subtracted.

    Labels are not decoration: hitlo.io.load_trigno_streams splits sides and
    segments by reading them and refuses to load a stream without them, rather
    than assuming a column order that would swap the legs.
    """
    labels = []
    for s in sensors:
        for axis in ("x", "y", "z"):
            labels.append(f"{s['label']}_acc_{axis}")
        for axis in ("x", "y", "z"):
            labels.append(f"{s['label']}_gyr_{axis}")

    info = StreamInfo("TrignoIMU", "IMU", len(labels), SAMPLE_RATE,
                      "float32", source_id)
    desc = info.desc()
    chans = desc.append_child("channels")
    for s in sensors:
        b = bias.get(s["slot"], [0.0, 0.0, 0.0])
        for axis in ("x", "y", "z"):
            ch = chans.append_child("channel")
            ch.append_child_value("label", f"{s['label']}_acc_{axis}")
            ch.append_child_value("unit", "g")
            ch.append_child_value("type", "ACC")
            ch.append_child_value("gyro_bias_subtracted", "0")
        for i, axis in enumerate(("x", "y", "z")):
            ch = chans.append_child("channel")
            ch.append_child_value("label", f"{s['label']}_gyr_{axis}")
            ch.append_child_value("unit", "deg/s")
            ch.append_child_value("type", "GYR")
            ch.append_child_value("gyro_bias_subtracted", f"{b[i]:.6f}")

    # Also as one block, so the correction is readable without walking every
    # channel element.
    acq = desc.append_child("acquisition")
    acq.append_child_value("gyro_bias_units", "deg/s")
    acq.append_child_value("accel_bias_corrected", "false")
    bias_node = acq.append_child("gyro_bias")
    for s in sensors:
        b = bias.get(s["slot"], [0.0, 0.0, 0.0])
        node = bias_node.append_child("sensor")
        node.append_child_value("label", s["label"])
        node.append_child_value("slot", str(s["slot"]))
        node.append_child_value("serial", str(s.get("serial", "")))
        node.append_child_value("x", f"{b[0]:.6f}")
        node.append_child_value("y", f"{b[1]:.6f}")
        node.append_child_value("z", f"{b[2]:.6f}")

    return StreamOutlet(info), labels


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--list", action="store_true",
                    help="print slot, serial, mode and channel range, then exit")
    ap.add_argument("--no-identify", action="store_true",
                    help="skip the shake test and use SENSOR_MAP")
    ap.add_argument("--no-bias", action="store_true",
                    help="skip the gyro bias measurement (publishes raw gyro)")
    ap.add_argument("--bias-seconds", type=float, default=5.0)
    args = ap.parse_args()

    cmd = socket.create_connection((args.host, CMD_PORT), timeout=5)
    cmd.recv(1024)                      # banner
    print(f"connected to base station at {args.host}:{CMD_PORT}")

    sensors = discover_sensors(cmd)
    if not sensors:
        print("no paired sensors found - pair them in the Trigno Control Utility")
        return 1

    print(f"\n{len(sensors)} sensor(s) paired:")
    for s in sensors:
        last = s["base"] + CHANS_PER_SENSOR - 1
        print(f"  slot {s['slot']:>2}  serial {s['serial']:<12} mode {s['mode']:<5} "
              f"STARTINDEX {s['startindex']}  channels {s['base']}-{last}")
    warn_about_modes(sensors)

    if args.list:
        return 0

    for s in sensors:
        s["label"] = SENSOR_MAP.get(s["slot"], f"slot{s['slot']}")

    unmapped = [s["slot"] for s in sensors if s["slot"] not in SENSOR_MAP]
    missing = [k for k in SENSOR_MAP if k not in {s["slot"] for s in sensors}]
    if unmapped:
        print(f"  note: slot(s) {unmapped} are paired but not in SENSOR_MAP")
    if missing:
        print(f"  note: slot(s) {missing} are in SENSOR_MAP but not paired")

    data = socket.create_connection((args.host, AUX_PORT), timeout=5)
    reader = FrameReader(data)
    command(cmd, "START")
    print("\nstreaming started")

    if not args.no_identify:
        assigned = identify_sides(reader, sensors,
                                  labels_wanted=list(SENSOR_MAP.values()))
        if assigned is None:
            print("  skipped - using SENSOR_MAP")
        else:
            for s in sensors:
                if s["slot"] in assigned:
                    s["label"] = assigned[s["slot"]]

    bias = {}
    if not args.no_bias:
        bias = measure_gyro_bias(reader, sensors, seconds=args.bias_seconds)

    outlet, labels = build_outlet(sensors, bias, source_id=f"trigno-{args.host}")
    print(f"\nLSL stream 'TrignoIMU' open, {len(labels)} channels:")
    for s in sensors:
        print(f"  slot {s['slot']:>2} -> {s['label']}")

    sanity = AccelSanityCheck(sensors)
    # Flat per-channel bias in published order, so the hot loop is a subtraction
    # against an index rather than a lookup.
    flat_bias = []
    for s in sensors:
        flat_bias.extend([0.0, 0.0, 0.0])
        flat_bias.extend(bias.get(s["slot"], [0.0, 0.0, 0.0]))

    n_pub = len(labels)
    sent = 0
    t_report = time.time()
    try:
        while True:
            frame = reader.next_frame()
            if frame is None:
                print("\ndata socket closed by base station")
                break
            sanity.feed(frame)

            sample = []
            for s in sensors:
                sample.extend(sensor_slice(frame, s["base"]))
            for i in range(n_pub):
                sample[i] -= flat_bias[i]

            outlet.push_sample(sample, local_clock())
            sent += 1
            if time.time() - t_report >= 10.0:
                print(f"  {sent} samples pushed", end="\r", flush=True)
                t_report = time.time()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        try:
            command(cmd, "STOP")
        except OSError:
            pass
        data.close()
        cmd.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
