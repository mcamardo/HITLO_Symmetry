# Sensor bridges

Scripts that get hardware onto LSL. Everything downstream consumes LSL, so a
bridge is the only place that knows about a vendor SDK or wire protocol.

| bridge | hardware | stream | rate |
|---|---|---|---|
| `../collect_sensors.py` | Polar H10 over BLE | `polar accel left` / `polar accel right`, one per side | ~200 Hz |
| `trigno_lsl_bridge.py` | Delsys Trigno Avanti via Trigno Control Utility | `TrignoIMU`, one stream for all sensors | ~148 Hz |

Full hardware walkthrough — sensor modes, channel allocation, slot mapping and
the failure modes that silently scramble body parts — is in
[docs/trigno_setup.md](../../docs/trigno_setup.md).

## The Trigno bridge

Runs on the Windows base-station machine, not here. `trigno_lsl_bridge.py` in
this directory is the version under source control; keep the copy on that
machine in step with it.

Three startup steps happen before the outlet opens, in this order, because
each depends on the one before:

1. **Side identification** — shake one sensor at a time and the bridge assigns
   the label from what actually moves. `SENSOR_MAP` is the fallback only.
   Slot numbers shift whenever sensors are paired or re-moded, and a stale map
   inverts the symmetry index while producing entirely plausible numbers.
2. **Gyro bias** — five seconds of quiet standing, per-axis mean subtracted
   from every published gyro sample and written into the stream metadata so
   the XDF records what was taken off. Accelerometers are left alone; their
   offset is not separable from gravity.
3. **Accelerometer sanity** — mean magnitude over the first second, warned once
   if any sensor sits outside 0.85–1.15 g.

Zero-filled frames arrive right after `START` and are skipped by all three
rather than averaged in.

For the repo to consume the stream unchanged, the LSL outlet needs to satisfy
two things:

**1. Declare channel labels.** `hitlo.io.load_trigno_streams` splits left from
right by reading them, and **refuses to load if they are missing** rather than
assuming a column order. A wrong assumption there would swap the legs, which
inverts the sign of the symmetry index while producing entirely plausible
numbers.

Labels carry side, body segment and modality:

```
left_shank_acc_x   left_shank_gyr_z    right_shank_acc_y   right_shank_gyr_x
right_foot_acc_x   right_foot_gyr_z    (foot/thigh optional)
```

A label with no segment (`left_acc_x`) is read as **shank**, so recordings made
before segments existed still load. Extra segments are carried through for
offline analysis; the cost function uses the shanks only.

`acc`/`accel` and `gyr`/`gyro` are both accepted; matching is
case-insensitive and column order does not matter.

In pylsl:

```python
info = StreamInfo('TrignoIMU', 'IMU', n_ch, 148, 'float32', source_id)
chans = info.desc().append_child('channels')
for label in labels:
    ch = chans.append_child('channel')
    ch.append_child_value('label', label)
    ch.append_child_value('unit', 'g' if 'acc' in label else 'deg/s')
    ch.append_child_value('type', 'ACC' if 'acc' in label else 'GYR')
```

**2. Use `float32`, not `int16`.** The Polar bridge declares `int16`, which
caused a silent failure: squaring it overflowed, `sqrt` of the wrapped value
gave NaN, and every recording produced zero heel strikes with no error. The
loader now casts defensively, but there is no reason to re-enter that hole.

## Checking a bridge before trusting it

```bash
../preflight.py                       # streams present and carrying samples
../compare_detectors.py <file.xdf>    # both detectors over one recording
```

If `load_trigno_streams` returns nothing, the usual cause is missing or
mismatched channel labels — the loader will not guess.
