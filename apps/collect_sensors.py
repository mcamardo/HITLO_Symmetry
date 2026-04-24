"""
apps/collect_sensors.py — connect one Polar H10 sensor and stream to LSL.

Usage (run TWO terminals, one per side):

    python apps/collect_sensors.py left
    python apps/collect_sensors.py right

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

SENSOR_IDS = {
    'left':  '7F302C25',
    'right': '80AE3629',
}

# Polar H10 BLE service / characteristic UUIDs
PMD_CONTROL_UUID = 'FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8'
PMD_DATA_UUID    = 'FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8'

# Request ACC @ 200 Hz, ±8 g, 16-bit signed
ACC_WRITE = bytearray([
    0x02, 0x02,
    0x00, 0x01, 0xC8, 0x00,   # sample rate 200 Hz
    0x01, 0x01, 0x10, 0x00,   # resolution 16-bit
    0x02, 0x01, 0x08, 0x00,   # range ±8 g
    0x04, 0x01, 0x03,         # channels = 3 (x,y,z)
])


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Connect a Polar H10 and stream accelerometer to LSL.")
    p.add_argument('side', choices=list(SENSOR_IDS.keys()),
                   help="Which shank: left or right")
    p.add_argument('--scan-timeout', type=float, default=15.0,
                   help="BLE scan timeout in seconds")
    return p.parse_args()


# ===========================================================================
# Helpers
# ===========================================================================

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

async def stream_sensor(side: str, scan_timeout: float) -> None:
    device_id = SENSOR_IDS[side]
    print(f"\n=== Starting Polar H10 ({side}) — ID: {device_id} ===\n")

    device = await find_sensor(device_id, timeout=scan_timeout)
    if device is None:
        print(f"\n💡 Tips:")
        print(f"   - Is the Polar strap wet and on a body? It won't advertise otherwise.")
        print(f"   - Is another BLE app connected? Close LabRecorder, browsers, etc.")
        print(f"   - Is the correct ID in SENSOR_IDS for '{side}' "
              f"(current: '{device_id}')?")
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

        # Request ACC stream + subscribe to data notifications
        await client.start_notify(PMD_DATA_UUID, handle_data)
        await client.write_gatt_char(PMD_CONTROL_UUID, ACC_WRITE, response=True)
        print(f"✅ ACC stream requested (200 Hz, ±8g)\n")

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
        asyncio.run(stream_sensor(args.side, args.scan_timeout))
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    return 0


if __name__ == '__main__':
    sys.exit(main())
