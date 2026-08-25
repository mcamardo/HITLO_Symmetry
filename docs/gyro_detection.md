# Gyroscope heel-strike detection

Used with shank-mounted Delsys Trigno Avanti IMUs. Selected by
`Sensing.detector: gyro`, which is the default when `backend: trigno`.

## The rule

Sagittal shank angular velocity has one large positive peak during swing as
the shank rotates forward, then reverses sharply as the foot meets the
ground. **Initial contact is the negative-going zero crossing immediately
after that peak.**

```
   swing peak
       ╱╲
      ╱  ╲
─────╱────╲──────────────  0
           ╲  ← initial contact (zero crossing)
            ╲___╱
          stance lobe
```

Three stages, and that is the whole method:

1. **Orient.** Lowpass at 15 Hz, then fix polarity (below).
2. **Find swing peaks.** Threshold at 35% of the trial's own 95th percentile
   — a fraction rather than an absolute deg/s, so it transfers across
   subjects and speeds without retuning. One peak per stride.
3. **Take the crossing after each peak.** Interpolated between samples,
   accepted only if the signal genuinely reverses afterward.

## Why this replaced the accelerometer method

The accelerometer detector finds an impact *transient* and must then choose
which of several similar peaks represents contact. That choice is what fails
when the shock is damped — a compliant mount, cushioned footwear, an
exoskeleton in the load path.

Measured across the final Polar sessions, the margin by which the accepted
peak beat its rejected competitors:

| leg | margin |
|---|---|
| free | 4.4 SD |
| instrumented | 0.16 SD |

At 0.16 the detector is choosing near-arbitrarily. A zero crossing is a sign
change, so there is nothing to choose between and no threshold that damping
can push the signal beneath. There is correspondingly no clustering, no
stance confirmation and no recovery pass — those exist only to disambiguate
competing peaks.

## Two things that needed care

**Polarity is resolved by trying both.** Gyro sign depends on how the sensor
was clipped to the shank. Getting it wrong locks onto the stance reversal
instead of swing, so every event lands at the wrong point in the cycle *while
still looking like a clean periodic detection*. Inferring it from which
excursion is larger fails when the two lobes are comparable — it decides on
noise. The detector runs both orientations and keeps the one with more
events, breaking ties on periodicity. Set `swing_sign` explicitly to pin it.

**The crossing is interpolated between samples.** At 148 Hz one sample is
6.8 ms. Symmetry index moves 0.4/stride points per millisecond — about 0.33
at a 1.2 s stride — so snapping to the nearest sample would inject roughly
two points of noise for nothing.

## Validation

`tests/test_regression.py` checks the detector against synthetic shank gyro
with known contact times across walking speed, noise, damped swing, 148 and
200 Hz, both mounting polarities, and symmetric/asymmetric lobes. Every
contact is recovered; worst median timing error 4.0 ms.

A further test pins that residual filter delay is **common-mode**. Step time
is a difference between legs, so an offset shared by both cancels exactly and
only a differential can reach the symmetry index. If that ever stops holding,
the detector would be injecting asymmetry that looks like gait.

## Comparing against the accelerometer

```bash
./apps/compare_detectors.py <file.xdf> --plot compare.png
```

Runs both methods over one recording and reports the per-leg timing offset.
A shared offset is harmless; a differential one goes straight into SI. Needs
a recording with both modalities, so Trigno only.

## Tunables

`GyroDetectionConfig` in `hitlo/detection_gyro.py`:

| field | default | what it does |
|---|---|---|
| `fs` | 148 | sample rate — use `for_stream()`, do not trust the default |
| `sagittal_axis` | 2 | which gyro column is the sagittal axis |
| `swing_sign` | None | pin polarity; None resolves it automatically |
| `lowpass_hz` | 15.0 | noise cleanup, well above swing and contact bandwidth |
| `swing_peak_frac` | 0.35 | peak threshold as a fraction of the trial's p95 |
| `min_peak_dist_s` | 0.40 | one swing peak per stride |
| `max_search_s` | 0.50 | how far after a peak to look for the crossing |
| `min_reversal_frac` | 0.08 | how far the signal must reverse to count |
| `reversal_window_s` | 0.15 | window in which to require that reversal |
