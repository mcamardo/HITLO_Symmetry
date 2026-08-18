#!/usr/bin/env python3.12
"""
apps/collect_sensors.py — connect one Polar H10 sensor and stream to LSL.

Usage (run TWO terminals, one per side):

    python apps/collect_sensors.py left
    python apps/collect_sensors.py right

Sensor IDs change between sessions (different straps). They are NOT hardcoded:

    python apps/collect_sensors.py --scan      # list every Polar in range
    python apps/collect_sensors.py --assign    # pick which strap is L / R

--assign writes config/sensors.json, which both terminals then read. Use
--id XXXXXXXX for a one-off override without touching the file.

After both terminals are streaming, CONFIRM the sides physically:

    python apps/verify_sides.py                # guided shake test over LSL

Reading the ID off the strap tells you which sensor is which. It does NOT
tell you which leg it ended up on. verify_sides.py is what proves that.

Each instance:
  1. Scans for the Polar H10 sensor whose Device ID matches config
  2. Connects over BLE
  3. Subscribes to the accelerometer PMD characteristic (200 Hz tri-axial)
  4. Re-streams samples into LSL as 'polar accel left' or 'polar accel right'

Why scan by device ID (not MAC address):
  Polar H10 MACs are randomized at each boot. Scanning by the device name
  ending lets us reconnect reliably across sessions without manual config
  changes. Your sensor IDs must match what's printed on the Polar strap itself
  (the 8-char suffix, e.g. '7F302C25' — left; '80AE3629' — right).

If you get "sensor not found" errors:
  - Make sure the Polar strap is wet against skin (it needs contact to boot)
  - Make sure no other app (including another terminal) is holding the BLE
    connection — only one client can own it at a time
  - Try moving closer to the laptop; BLE range is ~3 m through bodies
"""

import argparse
import asyncio
import json
import struct
import sys
from pathlib import Path

from bleak import BleakClient, BleakScanner
from pylsl import StreamInfo, StreamOutlet

# Make hitlo package importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ===========================================================================
# Sensor config — edit these to your own Polar IDs if they differ
# ===========================================================================

# Last known IDs, used only when config/sensors.json is absent. These are
# straps, not fixtures — expect them to change. Prefer --assign.
FALLBACK_SENSOR_IDS = {
    'left':  '7F302C25',
    'right': '80AE3629',
}

SENSOR_CONFIG_PATH = REPO_ROOT / 'config' / 'sensors.json'


def load_sensor_ids() -> dict:
    """Read the L/R assignment, falling back to the built-in defaults."""
    if SENSOR_CONFIG_PATH.is_file():
        try:
            data = json.loads(SENSOR_CONFIG_PATH.read_text())
            ids = {s: str(data[s]).strip().upper() for s in ('left', 'right')}
            if ids['left'] == ids['right']:
                print(f"config/sensors.json has the SAME id for both sides "
                      f"({ids['left']}). Re-run with --assign.")
                sys.exit(2)
            return ids
        except SystemExit:
            raise
        except Exception as e:
            print(f"Could not read {SENSOR_CONFIG_PATH} ({e}); "
                  f"using built-in defaults.")
    return dict(FALLBACK_SENSOR_IDS)


def save_sensor_ids(ids: dict) -> None:
    SENSOR_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SENSOR_CONFIG_PATH.write_text(json.dumps(ids, indent=2) + "\n")
    print(f"\nSaved to {SENSOR_CONFIG_PATH}:")
    for side in ('left', 'right'):
        print(f"   {side:5s} = {ids[side]}")


SENSOR_IDS = load_sensor_ids()

# Polar H10 BLE service / characteristic UUIDs
PMD_CONTROL_UUID = 'FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8'
PMD_DATA_UUID    = 'FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8'

# Request ACC @ 200 Hz, ±8 g, 16-bit signed
#
# Do NOT add a channel-count setting (0x04, 0x01, 0x03) here. The H10 firmware
# rejects the whole request with ERROR INVALID PARAMETER (control response
# f0 02 02 05) if it is present — verified by sweeping payload variants against
# 7F302C25: identical request with the field SUCCEEDS without it and fails with
# it. Channel count for ACC is implicit. The failure is silent unless the
# control response is checked, which is why check_control_response exists below.
ACC_WRITE = bytearray([
    0x02, 0x02,
    0x00, 0x01, 0xC8, 0x00,   # sample rate 200 Hz
    0x01, 0x01, 0x10, 0x00,   # resolution 16-bit
    0x02, 0x01, 0x08, 0x00,   # range ±8 g
])

# PMD control-point error codes, for turning a silent failure into a loud one.
PMD_ERRORS = {
    0: 'SUCCESS', 1: 'invalid op code', 2: 'invalid measurement type',
    3: 'not supported', 4: 'invalid length', 5: 'invalid parameter',
    6: 'already in state', 7: 'invalid resolution', 8: 'invalid sample rate',
    9: 'invalid range', 10: 'invalid MTU', 11: 'invalid n channels',
    12: 'invalid state', 13: 'device in charger',
}


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Connect a Polar H10 and stream accelerometer to LSL.",
        epilog="Sensor IDs live in config/sensors.json — set them with --assign.")
    p.add_argument('side', nargs='?', choices=['left', 'right'],
                   help="Which shank: left or right")
    p.add_argument('--scan', action='store_true',
                   help="List every Polar in range and exit")
    p.add_argument('--assign', action='store_true',
                   help="Interactively pick which strap is left/right, then save")
    p.add_argument('--id', dest='device_id', default=None,
                   help="Use this 8-char device ID for this run only")
    p.add_argument('--scan-timeout', type=float, default=15.0,
                   help="BLE scan timeout in seconds")
    a = p.parse_args()
    if not (a.scan or a.assign) and a.side is None:
        p.error("give a side (left/right), or use --scan / --assign")
    return a


# ===========================================================================
# Helpers
# ===========================================================================

def _device_id(name: str) -> str:
    """Polar advertises as e.g. 'Polar H10 7F302C25' — take the trailing token."""
    return (name or '').strip().split()[-1].upper() if name else ''


async def scan_polar(timeout: float) -> list:
    """Return [(device_id, name, address, rssi)] for every Polar in range."""
    print(f"Scanning for Polar devices (timeout {timeout}s) ...")
    rssi_by_addr = {}
    try:
        found = await BleakScanner.discover(timeout=timeout, return_adv=True)
        devices = []
        for addr, (dev, adv) in found.items():
            devices.append(dev)
            rssi_by_addr[addr] = getattr(adv, 'rssi', None)
    except TypeError:
        devices = await BleakScanner.discover(timeout=timeout)

    out = []
    for d in devices:
        if d.name and 'polar' in d.name.lower():
            out.append((_device_id(d.name), d.name, d.address,
                        rssi_by_addr.get(d.address)))
    out.sort(key=lambda r: (-(r[3] if r[3] is not None else -999), r[0]))
    return out


def _print_polar_table(found: list) -> None:
    if not found:
        print("\n  No Polar devices found.")
        print("  - Strap must be WET and on a body before it advertises")
        print("  - Close anything else holding a BLE connection")
        return
    print(f"\n  {len(found)} Polar device(s):\n")
    print(f"    {'#':<3} {'DEVICE ID':<12} {'RSSI':>6}   NAME")
    print(f"    {'-'*3} {'-'*12} {'-'*6}   {'-'*24}")
    for i, (did, name, addr, rssi) in enumerate(found, 1):
        r = f"{rssi} dBm" if rssi is not None else "  n/a"
        print(f"    {i:<3} {did:<12} {r:>6}   {name}")
    print("\n  Higher RSSI = closer to the laptop. Match the DEVICE ID against")
    print("  the 8 characters printed on each strap.")


async def cmd_scan(timeout: float) -> int:
    found = await scan_polar(timeout)
    _print_polar_table(found)
    if found:
        current = load_sensor_ids()
        print(f"\n  Currently configured:  left={current['left']}  "
              f"right={current['right']}")
        ids = {d[0] for d in found}
        for side in ('left', 'right'):
            if current[side] not in ids:
                print(f"  WARNING: configured {side} id {current[side]} "
                      f"is NOT in range.")
    return 0


async def cmd_assign(timeout: float) -> int:
    found = await scan_polar(timeout)
    _print_polar_table(found)
    if len(found) < 2:
        print("\nNeed at least 2 Polar devices in range to assign sides.")
        return 1

    def pick(side: str, taken: str = None) -> str:
        while True:
            raw = input(f"\n  Which # is on the {side.upper()} shank? ").strip()
            if not raw.isdigit() or not (1 <= int(raw) <= len(found)):
                print(f"  Enter a number 1-{len(found)}.")
                continue
            did = found[int(raw) - 1][0]
            if did == taken:
                print(f"  {did} is already assigned to the other side.")
                continue
            return did

    left = pick('left')
    right = pick('right', taken=left)
    save_sensor_ids({'left': left, 'right': right})
    print("\n  This records which STRAP is which, from the printed ID.")
    print("  It does not prove which LEG each ended up on. After both")
    print("  terminals are streaming, confirm that with:")
    print("      python apps/verify_sides.py")
    return 0


async def find_sensor(device_id: str, timeout: float):
    """Scan for the Polar H10 whose name ends with the device_id (8 char)."""
    print(f"🔍 Scanning for Polar H10 with ID ending in {device_id} "
          f"(timeout {timeout}s) ...")
    devices = await BleakScanner.discover(timeout=timeout)
    for d in devices:
        if d.name and d.name.endswith(device_id):
            print(f"✅ Found: {d.name}  (address={d.address})")
            return d
    print(f"❌ No device found with ID {device_id}")
    print(f"   Scanned devices:")
    for d in devices:
        if d.name:
            print(f"      - {d.name}")
    return None


def parse_acc_frame(data: bytes):
    """Parse a Polar PMD accelerometer frame into samples.

    Frame layout:
        byte 0   : measurement type (0x02 = ACC)
        bytes 1-8: reference LSL timestamp (we ignore, LSL outlet stamps below)
        byte 9   : frame type (0x01 = 16-bit accel)
        bytes 10+ : repeating (x_i16, y_i16, z_i16)

    Returns list of (x, y, z) tuples in mG.
    """
    samples = []
    if len(data) < 10 or data[0] != 0x02:
        return samples
    frame_type = data[9]
    if frame_type != 0x01:
        return samples
    offset = 10
    while offset + 6 <= len(data):
        x, y, z = struct.unpack_from('<hhh', data, offset)
        samples.append((x, y, z))
        offset += 6
    return samples


# ===========================================================================
# Main stream loop
# ===========================================================================

async def stream_sensor(side: str, scan_timeout: float,
                        device_id: str = None) -> None:
    device_id = (device_id or SENSOR_IDS[side]).strip().upper()
    print(f"\n=== Starting Polar H10 ({side}) — ID: {device_id} ===\n")

    device = await find_sensor(device_id, timeout=scan_timeout)
    if device is None:
        print(f"\n💡 Tips:")
        print(f"   - Is the Polar strap wet and on a body? It won't advertise otherwise.")
        print(f"   - Is another BLE app connected? Close LabRecorder, browsers, etc.")
        print(f"   - Is '{device_id}' the right ID for '{side}'? "
              f"Run --scan to see what is actually in range,")
        print(f"     then --assign to set it.")
        return

    print(f"\n🔗 Connecting to {device.address} ...")
    async with BleakClient(device.address) as client:
        if not client.is_connected:
            print(f"❌ BleakClient reports disconnected.")
            return
        print(f"✅ connected")

        # LSL outlet
        stream_name = f'polar accel {side}'
        info = StreamInfo(
            name=stream_name, type='ACC', channel_count=3,
            nominal_srate=200, channel_format='int16',
            source_id=f'polar_h10_{device_id}',
        )
        chns = info.desc().append_child('channels')
        for label in ['X', 'Y', 'Z']:
            ch = chns.append_child('channel')
            ch.append_child_value('label', label)
            ch.append_child_value('unit', 'mG')
            ch.append_child_value('type', 'ACC')
        outlet = StreamOutlet(info, chunk_size=1)
        print(f"📡 LSL outlet opened: name='{stream_name}'")
        print(f"   Ready for LabRecorder — click Update and it should appear.\n")

        # Sample counter (for status prints every ~10 s)
        n_samples = 0
        next_print_samples = 200 * 10

        def handle_data(_sender, data):
            nonlocal n_samples, next_print_samples
            for (x, y, z) in parse_acc_frame(data):
                outlet.push_sample([x, y, z])
                n_samples += 1
            if n_samples >= next_print_samples:
                print(f"   ... streaming ({n_samples} samples pushed)")
                next_print_samples += 200 * 10

        # Watch the control point so a rejected configuration is visible.
        # Without this the sensor can refuse the settings and the script still
        # reports success, leaving an open LSL outlet that never carries data.
        ctrl_status = {}

        def handle_ctrl(_sender, data):
            if len(data) > 3 and data[0] == 0xF0:
                ctrl_status['code'] = data[3]
                ctrl_status['hex'] = data.hex()

        await client.start_notify(PMD_CONTROL_UUID, handle_ctrl)
        await client.start_notify(PMD_DATA_UUID, handle_data)
        await client.write_gatt_char(PMD_CONTROL_UUID, ACC_WRITE, response=True)

        await asyncio.sleep(1.0)
        code = ctrl_status.get('code')
        if code is None:
            print("⚠️  No control response from the sensor — cannot confirm the "
                  "ACC stream started.")
        elif code != 0:
            print(f"❌ Sensor REJECTED the ACC configuration: "
                  f"{PMD_ERRORS.get(code, f'code {code}')} "
                  f"(response {ctrl_status.get('hex')})")
            print(f"   No data will arrive. Not leaving a dead outlet open.")
            await client.stop_notify(PMD_DATA_UUID)
            return
        else:
            print(f"✅ ACC stream confirmed by sensor (200 Hz, ±8g)")

        # Confirm real samples, not just a successful handshake.
        await asyncio.sleep(2.0)
        if n_samples == 0:
            print("⚠️  Handshake succeeded but ZERO samples arrived in 2s. "
                  "Check electrode contact.")
        else:
            print(f"✅ {n_samples} samples in first 2s "
                  f"(~{n_samples / 2.0:.0f} Hz)\n")

        # Keep running until Ctrl-C
        try:
            while True:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            await client.stop_notify(PMD_DATA_UUID)
            print(f"\n🛑 Stopped. Total samples: {n_samples}")


def main() -> int:
    args = parse_args()
    try:
        if args.scan:
            return asyncio.run(cmd_scan(args.scan_timeout))
        if args.assign:
            return asyncio.run(cmd_assign(args.scan_timeout))
        asyncio.run(stream_sensor(args.side, args.scan_timeout,
                                  device_id=args.device_id))
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    return 0


if __name__ == '__main__':
    sys.exit(main())
