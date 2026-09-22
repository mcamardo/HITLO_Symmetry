# Shank IMU data: a primer

This is an introduction to the inertial measurement unit (IMU) data collected
on the shanks, written for someone who has just been handed a sample dataset
and has not worked with this repo before. By the end you should be able to
open a trial, find the heel strikes, compute a symmetry index, and tell
whether the result is trustworthy.

Work through it in order. The code examples build on each other, and every one
of them has been run against a real recording.

**Scope note.** This repo also contains an older Polar accelerometer backend.
Ignore it. If you see anything referring to `polar accel left`, `polar accel
right`, or a sternum-mounted sensor, it does not apply to the data you have.
Everything below is the Trigno shank IMU path.

**Two detectors, one current.** The file `docs/detection_pipeline.md`
describes an accelerometer and jerk pipeline. That was the original method and
it is no longer the one in use. The production config uses `detector: 'gyro'`,
which finds heel strikes in the gyroscope signal instead. Read
`docs/gyro_detection.md` for the current method. The accelerometer pipeline is
worth understanding as background, and there is a section below on why it was
not sufficient, but do not use it for new analysis.

---

## 1. The hardware

Two Delsys Trigno Avanti IMUs, one on each shank. Each sensor carries a
tri-axial accelerometer and a tri-axial gyroscope. Both are streamed over Lab
Streaming Layer (LSL) and recorded into a single XDF file per trial, at about
148 Hz.

Points that matter for the analysis:

| Fact | Consequence |
| --- | --- |
| The sensors are held on with Coban wrap | The mount is somewhat compliant, so impact shocks are damped |
| Mounting orientation is not controlled | Which gyro axis is the sagittal one differs between sessions, and so does its sign |
| Both sensors land in one LSL stream | The side is encoded in the channel label, not in the stream name |
| Sample rate is about 148 Hz | One sample is about 6.8 ms, which is coarse for an impact and fine for a rotation |

The uncontrolled mounting orientation is the single most important thing on
that list. Nothing in the pipeline assumes the sensor is the right way up or
the right way round. Section 3 explains how it works that out from the data.

---

## 2. What is in an XDF file

### 2.1 File naming

Recordings follow a BIDS-style convention:

```
sub-P091_ses-S001_task-Pre_run-001_motion.xdf
```

Broken down:

| Part | Meaning |
| --- | --- |
| `sub-P091` | Subject identifier |
| `ses-S001` | Session |
| `task-Pre` | Which phase of the protocol |
| `run-001` | Trial number within that phase |
| `motion` | BIDS modality for IMU data |

The task tag tells you what the trial was for:

| Task | What it is |
| --- | --- |
| `Pre` | Baseline, recorded before optimization. `run-001` is familiarization and is ignored. `run-002` is the trial whose symmetry index defines the subject's baseline. |
| `Default` | An optimization trial. One per device setting the optimizer tried. |
| `Post` | Recorded after optimization, to see what carried over. |

Files live under `~/HITLO_Data/sub-P0XX/ses-S001/motion/`.

### 2.2 Loading a trial

There is one LSL stream in the file, named `TrignoIMU`, holding every channel
from both sensors. The loader splits it into two per-limb objects by reading
the channel labels. You do not need to do that yourself.

```python
from hitlo.io import load_streams

XDF = ('/Users/you/HITLO_Data/sub-P091/ses-S001/motion/'
       'sub-P091_ses-S001_task-Pre_run-001_motion.xdf')

cfg = {'Sensing': {'backend': 'trigno', 'detector': 'gyro'}}
left, right = load_streams(XDF, cfg)

if left is None or right is None:
    raise SystemExit('one or both shank streams are missing from this file')
```

Always pass that `cfg` dictionary. `load_streams` defaults to the old Polar
behaviour when it is not given one, and will return `(None, None)` on a Trigno
file. The same dictionary selects the gyro detector later, so define it once
at the top of your script and reuse it.

`load_streams` returns `(left, right)`. Either can be `None`, which is why the
guard above is there. A side comes back as `None` when its channels are absent
or incomplete. The loader refuses to guess at column order, because a wrong
guess would silently swap left and right, and that flips the sign of the final
result.

### 2.3 What you get back

Each side is a `SensorStream`. Its fields:

| Field | Type | Notes |
| --- | --- | --- |
| `accel` | ndarray, shape (N, 3) | Accelerometer, in g |
| `gyro` | ndarray, shape (N, 3), or `None` | Gyroscope, in deg/s |
| `timestamps` | ndarray, shape (N,) | LSL seconds. Not zero-based. |
| `actual_fs` | float | Rate measured from the timestamps, not the nominal one |
| `side` | str | `'left'` or `'right'` |
| `name` | str | Source, for example `'TrignoIMU:left'` |
| `backend` | str | `'trigno'` here |
| `has_gyro` | property | `True` when gyro data is present and the right length |

On a real trial that prints:

```
left.accel      (17854, 3)
left.gyro       (17854, 3)
left.timestamps (17854,)  first 2421606.259
left.actual_fs  148.15
left.side       'left'   left.name 'TrignoIMU:left'
left.backend    'trigno'  has_gyro True
```

Two things to notice. The timestamps start at 2421606, not at zero, because
LSL counts seconds from an arbitrary epoch. Subtract the first timestamp when
you want a plot that starts at zero. And `actual_fs` is 148.15, not exactly
148. Use the measured value, never the nominal one. Every detector window is
specified in seconds and converted using the sample rate, so a wrong rate
moves every window.

To see which sensors a file contains before loading it:

```python
from hitlo.io import trigno_inventory
print(trigno_inventory(XDF))     # {'left': ['shank'], 'right': ['shank']}
```

Some sessions also carry foot sensors. This primer covers shanks only.

---

## 3. Finding heel strikes with the gyroscope

### 3.1 The idea

During walking the shank rotates forward quickly through swing. That produces
one large, unmistakable peak in angular velocity. The rotation then reverses
sharply when the foot hits the ground. Initial contact is the negative-going
**zero crossing** immediately after the swing peak.

A zero crossing is a sign change. There is no threshold to tune and no choice
between competing peaks, which is what makes this robust. See
`docs/gyro_detection.md` for the full rule and its validation.

### 3.2 Axis selection, in two steps

Because mounting is not controlled, the detector has to work out two things
before it can apply that rule.

**Step one: which axis is the sagittal one.** During walking, the axis the leg
actually rotates about carries far more variance than the other two. So the
detector takes the standard deviation of each gyro column and picks the
largest.

```python
import numpy as np

sd = left.gyro.std(axis=0)
order = np.argsort(sd)[::-1]
axis = int(order[0])
dominance = sd[order[0]] / sd[order[1]]

print(f"per-axis sd (deg/s): x={sd[0]:.1f} y={sd[1]:.1f} z={sd[2]:.1f}")
print(f"sagittal axis = {'xyz'[axis]}   dominance = {dominance:.2f}x")
```

On the trial above:

```
per-axis sd (deg/s): x=21.8 y=52.8 z=157.6
sagittal axis = z   dominance = 2.99x
```

The ratio between the winner and the runner-up is called **dominance**, and it
is a quality measure for the mounting. At 2.99x this sensor is clearly
mounted. Below about 1.6x the two largest axes are close enough that the
choice can flip between two trials of the same subject on noise alone, and the
detector then reads a different physical rotation in each. The detector warns
when dominance falls under 1.15x and falls back to the z axis. Treat anything
under 1.6x as a trial to check by eye.

**Step two: which sign is swing.** Mounting decides whether forward rotation
reads positive or negative. Mid-swing is the fastest rotation in the gait
cycle, so whichever polarity holds the larger excursion is swing. Flip the
trace so that lobe is positive.

```python
w = left.gyro[:, axis]
if abs(w.min()) > abs(w.max()):
    w = -w
print(f"largest positive {w.max():.0f}, largest negative {w.min():.0f}")
# largest positive 380, largest negative -213
```

If you skip this step and the sensor happens to be mounted the other way
round, the detector locks onto the stance reversal instead of the swing peak.
Every event then lands at the wrong point in the cycle, while still looking
like a clean, regular detection. That is the failure mode to fear here,
because nothing about the output looks wrong.

You do not have to write either step yourself. `detect` does both internally.
They are shown here so you know what it is doing and why the QC plot reports
an axis letter and a dominance number.

### 3.3 Running the detector

```python
from hitlo.detectors import detect

res_left = detect(left, cfg)
res_right = detect(right, cfg)

print(res_left.heel_strike_times[:3])   # [2421607.292 2421608.401 2421609.507]
print(len(res_left.heel_strike_times))  # 107
```

`detect` returns a `DetectionResult`. The fields you will use:

| Field | Meaning |
| --- | --- |
| `heel_strike_times` | Contact times in LSL seconds. This is the output that matters. |
| `heel_strike_indices` | Sample index of each contact |
| `all_candidates` | Every swing peak considered |
| `strict_peaks` | Swing peaks that produced a valid crossing |
| `rejected_peaks` | Swing peaks with no valid crossing after them |
| `recovered_peaks` | Always empty for the gyro detector |

The times are interpolated between samples rather than rounded to the nearest
one. At 148 Hz a sample is 6.8 ms, and the symmetry index moves about 0.28
points per millisecond of timing error, so rounding would inject roughly two
points of noise for nothing.

`recovered_peaks` is always empty because the recovery pass belongs to the
accelerometer pipeline, which needed it to disambiguate competing peaks. A
zero crossing is unique within a cycle, so there is nothing to recover.

If you ask for gyro detection on a stream with no gyro, `detect` raises a
`ValueError` rather than falling back to the accelerometer. That is
deliberate. The two methods find different instants, so a silent fallback
would produce a symmetry index that is not comparable with the rest of the
session.

---

## 4. Why not the accelerometer

The original detector looked for the impact shock: filter the acceleration
magnitude, differentiate to get jerk, threshold it, cluster the peaks, and
pick one per cluster. It works, but it is fragile here, for reasons that are
not tunable.

| Problem | Number |
| --- | --- |
| The impact is barely sampled | About 2 samples wide at 148 Hz |
| Its sampled height varies stride to stride | About 2.7x |
| A soft strike barely clears the background | About 1.2 g against a 1.0 g walking baseline |

No threshold separates those cases. The situation gets worse when the shock is
damped, which is exactly what a Coban-wrapped sensor and an exoskeleton in the
load path do. On two test subjects the accepted peak beat its rejected
competitors by 4.4 standard deviations on the free leg and only 0.16 on the
instrumented one. On the instrumented leg the detector was essentially
guessing.

The gyroscope does not have this problem because the swing rotation is large,
slow, and smooth, and because a sign change needs no amplitude threshold at
all.

One consequence worth remembering: **the two detectors find different
instants.** The gyro zero crossing is initial contact. The jerk peak is the
shock that follows it, tens of milliseconds later. Do not compare a symmetry
index from one against a baseline collected with the other.

---

## 5. Trimming the trial

A recording is not walking from end to end. The subject stands while the
operator sets the device up, walks, and then stops while the recorder is still
running. Those parts must come out before you compute anything.

This happens in two stages.

**Stage one, find the walking.** `walking_window` looks at the combined gyro
activity of the two shanks and returns the longest continuous stretch above a
threshold, as a pair of LSL timestamps.

```python
from hitlo.symmetry import walking_window

win = walking_window(left, right)
if win is None:
    t0 = min(left.timestamps[0], right.timestamps[0])
    t1 = max(left.timestamps[-1], right.timestamps[-1])
else:
    t0, t1 = win
print(f"window {t1 - t0:.1f} s, found={win is not None}")   # 120.3 s, found=True
```

Write that guard every time. `walking_window` returns `None` when there is no
gyro, or when nothing clears the threshold for at least 10 seconds. It returns
`None` rather than raising so a caller can fall back to the whole recording
instead of losing the trial, but that means an unguarded `t0, t1 = ...` will
crash with a confusing error on exactly the trials you most want to inspect.

**Stage two, drop the edge strides.** `trim_peaks` removes heel strikes within
`trim_s` seconds of each end of the window.

```python
from hitlo.symmetry import trim_peaks
import numpy as np

lt = trim_peaks(np.sort(res_left.heel_strike_times), t0, t1, 3.0)
rt = trim_peaks(np.sort(res_right.heel_strike_times), t0, t1, 3.0)
print(len(lt), len(rt))    # 103 102
```

Three seconds is the usual value. Edge strides are dropped because starting
and stopping have systematically different mechanics from steady walking.
Shank accelerations are weaker during startup, and when a perturbation is
first applied the first strides reflect a subject actively correcting rather
than the adapted state you are trying to measure. Leaving them in biases the
trial mean by an amount that varies with how briskly the subject got going,
which is not something you want in your data.

---

## 6. Step times and the symmetry index

### 6.1 Steps

A step is the gap from one foot's contact to the other foot's next contact.

| Term | Definition |
| --- | --- |
| Right step | Left heel strike, then the next right heel strike |
| Left step | Right heel strike, then the next left heel strike |
| Stride | One step plus the following step, so one full gait cycle |

`compute_step_times` interleaves the two lists and returns both.

```python
from hitlo.symmetry import compute_step_times

right_steps, left_steps = compute_step_times(lt, rt)
print(right_steps.mean(), left_steps.mean())   # 0.561 0.556
```

Note the return order. Right steps come first. It is easy to swap them by
accident, and a swap flips the sign of everything downstream.

Both lists must be on a common time base, which is why the pipeline uses LSL
timestamps throughout and never sample indices. The two sensors run at
slightly different actual rates, so sample index 1000 is not the same instant
on both.

### 6.2 The symmetry index

```
SI = 2 × (right step − left step) / (right step + left step) × 100 %
```

```python
from hitlo.symmetry import compute_symmetry_index

si, per_stride = compute_symmetry_index(right_steps, left_steps, signed=True)
print(f"SI = {si:+.2f}%   sd = {per_stride.std():.2f}")   # SI = +0.97%   sd = 1.73
```

It returns two things: the mean across strides, and the per-stride values. The
per-stride array is always signed, whatever you pass for `signed`. That flag
only affects the mean.

Interpretation:

| SI | Meaning |
| --- | --- |
| 0 | Symmetric. The two steps take the same time. |
| Greater than 0 | Right step is longer. The left leg spends longer in support. |
| Less than 0 | Left step is longer. The right leg spends longer in support. |
| About ±1 to ±2 | Within the normal range for a healthy walker |
| About −9 | Roughly what wearing the unpowered device did to one subject, for scale |

Keep `signed=True`. The direction is the measurement, not a detail. An
unsigned index cannot tell a subject who improved from one who overshot past
symmetry in the other direction.

---

## 7. The short way, and the QC plot

Everything in sections 2 to 6 is wrapped in one function.

```python
from hitlo.plot_heelstrikes import analyze_gyro

qc = analyze_gyro(XDF, 3.0)     # path, trim seconds
print(f"SI {qc['si']:+.2f}%   L {qc['n_left']} strikes   R {qc['n_right']} strikes")
# SI +0.97%   L 103 strikes   R 102 strikes
```

It returns a dictionary:

| Key | Contents |
| --- | --- |
| `si` | Signed symmetry index |
| `per` | Per-stride symmetry indices |
| `n_left`, `n_right` | Heel strike counts after the trim |
| `l_steps`, `r_steps` | Step times |
| `step_t` | Time of each step, for plotting |
| `win` | Walking window, relative to the start of the recording |
| `had_window` | `False` when `walking_window` found nothing and the whole trial was used |
| `left`, `right` | Per-side detail: `t`, `w` (oriented trace), `axis`, `flipped`, `dominance`, `hs` |
| `t0`, `trim` | Time origin and the trim that was applied |

Use it for anything routine. Write out the steps by hand when you are
debugging a trial that looks wrong, because then you can print the
intermediate values.

To see the result rather than just the number:

```python
from hitlo.plot_heelstrikes import make_gyro_plot

make_gyro_plot(qc)                                # opens a window
make_gyro_plot(qc, save_path='p091_pre1_qc.png')  # or saves a file
```

You can also run it from the command line without writing a script:

```
python hitlo/plot_heelstrikes.py <file.xdf> --detector gyro --trim 3.0 --save qc.png
```

The `--detector gyro` flag is required. The script still defaults to the old
accelerometer view.

The plot has three panels. The top two show each shank's oriented gyro trace
with the detected strikes marked, titled with the axis that was chosen and its
dominance. The bottom shows the step times the symmetry index was computed
from.

**Look at this plot for every trial.** Not a sample of them, every one. A
symmetry index is a single number summarizing a thousand events, and it will
happily report a plausible value for a detection that is completely wrong. The
plot is where wrong detection is visible and the number is where it is
invisible. It takes about five seconds per trial.

---

## 8. Sanity checks

Run through these before you believe a trial.

| Check | What you want | What a failure means |
| --- | --- | --- |
| Event counts | Left and right within one or two of each other | One leg's detector is finding events the other is not |
| Stride times | Both legs near the same value, typically 1.0 to 1.5 s | If one leg is close to half the other, that detector is counting each stride twice |
| The two traces | Similar in shape, both with swing positive | If one looks like the mirror image of the other, the sign inference went the wrong way on one side |
| Per-stride scatter | Standard deviation of a few points | A large value means the mean is not describing a steady gait |
| Dominance | Above 1.6x on both sides | Below that, the axis choice is near-arbitrary and can differ between trials |

The trial used throughout this primer passes all five: 103 against 102 events,
stride times of 1.118 s with a range of 1.082 to 1.162, per-stride standard
deviation of 1.73, dominance 2.99x.

For contrast, here is a real failure. One subject's trial produced a right
stride of 0.711 s against a left of 1.422 s, an exact 2 to 1 ratio. Each leg
looked perfectly plausible on its own, and every per-leg check passed. It
reached the optimizer as a symmetry index of +67.89%. Two legs of one person
walking share a cadence, so when they disagree it is the detector that is
wrong, not the participant. That is why the cross-leg checks exist.

Those checks are automated in `hitlo.symmetry.leg_consistency`, which returns
a list of human-readable warnings, and `hitlo.symmetry.axis_agreement`, which
checks whether the two shanks resolved to the same rotation. They warn, they
do not alter the symmetry index. Reading the warning is your job.

```python
from hitlo.symmetry import leg_consistency

for warning in leg_consistency(lt, rt, per_stride):
    print(warning)
```

---

## 9. First exercise

Reproduce the symmetry index by hand and check it against `analyze_gyro`.

1. Pick a trial from the sample dataset.
2. Load it with `load_streams` and the `cfg` dictionary from section 2.2.
3. Run `detect` on each side.
4. Get the walking window, with the `None` guard.
5. Trim both sets of heel strikes by 3 seconds.
6. Compute step times, then the symmetry index.
7. Call `analyze_gyro(path, 3.0)` on the same file and compare.

The two should agree to two decimal places. On the trial used here both give
`+0.97`, because `analyze_gyro` runs exactly these steps.

When they agree, you have the whole pipeline in your head. Then do three more
things:

- Make the QC plot and find the heel strikes visually in the top panel.
  Convince yourself the markers sit where the trace crosses zero going
  downward, just after each large peak.
- Print `sd` for each side and note the dominance. Compare a well-mounted
  trial against a poorly-mounted one if the sample dataset has both.
- Change the trim from 3 seconds to 0 and see how much the symmetry index
  moves. That tells you how much the edge strides were contributing.

If the numbers do not match, the usual causes are a forgotten `cfg`, sorting
one list of heel strikes but not the other, or swapping the return order of
`compute_step_times`.

---

## Where to go next

| Document | Covers |
| --- | --- |
| `docs/gyro_detection.md` | The detection rule in full, with its validation. Read this next. |
| `docs/trigno_setup.md` | Hardware, the LSL bridge, and recording a session |
| `docs/workflow.md` | How a session runs end to end |
| `docs/detection_pipeline.md` | The old accelerometer method. Background only. |
