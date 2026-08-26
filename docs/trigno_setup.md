# Setting up the Delsys Trigno

Getting Trigno Avanti IMUs onto LSL so this repo can read them. Written from
what actually worked on the hardware, including the two things that silently
break the channel mapping.

---

## The layout

Three pieces, and they are usually not the same machine:

```
Trigno sensors ──wireless──▶ Base station ──USB──▶ Windows PC
                                                   running Trigno Control
                                                   Utility (TCU), serving
                                                   TCP on 50040–50044
                                                        │
                                                   bridge script
                                                   reads AUX port 50044
                                                        │
                                                    LSL "TrignoIMU"
                                                        │  (network)
                                                 ┌──────┴──────┐
                                              Mac: LabRecorder + console
```

The bridge does not have to run on the same machine as this repo, and in our
setup it does not — TCU is Windows-only. **Nothing here filters LSL streams by
host**, so a bridge on another machine is expected and supported.

**Ports:** `50040`–`50042` command/EMG, `50043` EMG data, **`50044` AUX** —
which is where accelerometer and gyroscope live. The bridge reads AUX only.

---

## Step 1 — Put every sensor in the same mode

**Do this first. It is the single most common cause of scrambled channels.**

In TCU, check each paired sensor's mode. You want an IMU-only mode with **0 EMG
channels** (mode `609` on our units) for all of them.

Why it matters: the base station allocates channel blocks according to what
each sensor reports. One sensor in a mode with an EMG channel (e.g. `65`) is
allocated differently from its neighbours, which **shifts the channel offsets
of every sensor after it**. The data still streams and still looks plausible —
it is just attributed to the wrong body part.

A good bridge warns about mixed modes at startup. Fix it in TCU rather than
working around it.

---

## Step 2 — Learn where each sensor's channels are

Do not assume slot 1 is channels 0–5. The base station tells you, and the
bridge should ask it every time it starts.

Query each sensor's `STARTINDEX`, then:

```
channel_base = (STARTINDEX − 1) × 9
```

Nine channels of stride per sensor block; the first six are accel x/y/z then
gyro x/y/z. Worked example from our base station:

| slot | STARTINDEX | channel base | channels used |
|---|---|---|---|
| 1 | 1 | 0 | 0–5 |
| 3 | 2 | 9 | 9–14 |
| 5 | 4 | 27 | 27–32 |

Note slot 3 has `STARTINDEX` 2, not 3 — the index counts *paired* sensors, not
slot numbers. That is exactly why it must be queried rather than computed from
the slot.

**Two assumptions baked into that formula**, both worth knowing because a mode
change breaks them:

- **6 channels read per sensor.** Our sensors report `AUXCHANNELCOUNT` 6
  (accel + gyro, magnetometer off). A mode with the magnetometer on reports 9;
  the extra three are dropped, which is harmless.
- **A stride of 9.** Inferred from the allocation above, not from Delsys
  documentation. If you ever see one sensor's data appearing under another's
  label, question this first.

Querying at startup means unpairing a sensor, adding one, or changing a mode is
picked up automatically on the next launch. Nothing to re-run by hand.

---

## Step 3 — Map slots to body positions

One line you maintain, using the same slot numbers TCU shows:

```python
SENSOR_MAP = {
    1: "left_shank",
    3: "right_shank",
    5: "left_foot",
}
```

Edit it only when the physical arrangement changes — a different sensor in a
slot, or the same sensor moved to another body part. If sensors stay in the
same slots on the same body parts, you never touch it.

The bridge should warn when a mapped slot is not paired, and when something is
paired that is not mapped. That turns a sensor dying mid-session into a warning
rather than a silent relabelling.

**What no software can catch** is physically swapping two sensors between legs
without updating the map. Two defences:

1. Record the serial numbers next to each body position once (a `--list` mode
   printing slot, serial, mode and channel range is worth having), and check
   them at setup.
2. Run the shake test on the console's Sensors page, or `./apps/verify_sides.py`,
   before every session.

A swap inverts the sign of the symmetry index while producing entirely
plausible numbers. This has happened here — see *Known failure modes* below.

---

## Step 4 — Channel labels

This repo splits sides and body segments **by reading the labels**, and
`hitlo.io` refuses to load rather than guessing a column order. Label format:

```
<side>_<segment>_<modality>_<axis>

left_shank_acc_x    left_shank_gyr_z
right_shank_acc_y   right_foot_gyr_x
```

- `acc`/`accel` and `gyr`/`gyro` are both accepted, case-insensitive
- Column order does not matter
- A label without a segment (`left_acc_x`) is read as **shank**, so older
  recordings still load
- Extra segments (`foot`, `thigh`) are carried through and available to
  offline analysis, but the cost function uses the shanks only

Declare them as `float32`. See [../apps/bridges/README.md](../apps/bridges/README.md)
for the pylsl snippet and why `int16` caused a silent failure.

---

## Step 5 — Verify before you trust it

```bash
./apps/preflight.py            # stream present, rate, channel count, host
./apps/verify_sides.py         # which stream is which leg
```

Then in the console's **Sensors** page: check the inventory lists
`left_shank` and `right_shank`, and run the shake test.

After the first recording:

```bash
streamlit run apps/trial_explorer.py    # pick the file, look at the detection
```

---

## Known failure modes

**Mixed sensor modes.** One sensor with an EMG channel shifts every subsequent
sensor's channel offsets. Data looks fine, body parts are wrong. Fix in TCU.

**A foot sensor labelled for the wrong leg.** Happened here for a full session:
a sensor recorded as `left_foot` was physically on the right foot. Step-time
symmetry was unaffected (it uses shanks only), but any foot-based analysis
paired a foot with the opposite shank. `hitlo.ankle_angle.verify_foot_side()`
detects this from gait phase — a foot swings with the shank above it, half a
stride from the other — and the explorer flags it and offers the corrected
pairing.

**Assuming slot number equals STARTINDEX.** They differ as soon as a slot is
empty. Query it.

**EMG is not in this stream.** The bridge reads AUX (50044) only. EMG lives on
50043 and would need a second bridge publishing its own outlet; LabRecorder
records both into one XDF on a shared clock. EMG at 2000 Hz has no business in
the same stream as 148 Hz IMU.

---

## Rate

~148 Hz. Fast enough for the gyro heel-strike landmark, which spans about 39
samples, and marginal for the accelerometer impact, which spans about 2 — see
the "Which to prefer" note in the [README](../README.md#detection-pipelines).
