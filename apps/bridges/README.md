# Sensor bridges

Scripts that get hardware onto LSL. Everything downstream consumes LSL, so a
bridge is the only place that knows about a vendor SDK or wire protocol.

| bridge | hardware | stream | rate |
|---|---|---|---|
| `../collect_sensors.py` | Polar H10 over BLE | `polar accel left` / `polar accel right`, one per side | ~200 Hz |
| `trigno_bridge.py` *(to add)* | Delsys Trigno Avanti via Trigno Control Utility | `TrignoIMU`, one stream for all sensors | ~148 Hz |

## Adding the Trigno bridge

Drop the working script in here as `trigno_bridge.py`. For the repo to
consume it unchanged, the LSL outlet needs to satisfy two things:

**1. Declare channel labels.** `hitlo.io.load_trigno_streams` splits left from
right by reading them, and **refuses to load if they are missing** rather than
assuming a column order. A wrong assumption there would swap the legs, which
inverts the sign of the symmetry index while producing entirely plausible
numbers.

Labels must start with the side and contain the modality:

```
left_acc_x  left_acc_y  left_acc_z  left_gyr_x  left_gyr_y  left_gyr_z
right_acc_x right_acc_y right_acc_z right_gyr_x right_gyr_y right_gyr_z
```

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
