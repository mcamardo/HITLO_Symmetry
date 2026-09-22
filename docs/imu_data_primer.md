# Shank IMU data: a primer

You have a set of walking recordings from two leg-mounted sensors. This page
gets you from a file to a number, and tells you how to know whether the number
is any good. Allow an afternoon.

**Ignore the Polar backend.** This repo also supports an older chest-strap
sensor. If you see `polar accel left`, `polar accel right`, or anything about a
sternum-mounted sensor, it does not apply to your data.

**Use the gyro detector.** There are two ways to find heel strikes in this
repo. The old one uses the accelerometer and is described in
`docs/detection_pipeline.md`. The current one uses the gyroscope and is
described in `docs/gyro_detection.md`. Everything here uses the gyroscope.

---

## 1. What you have

Two Delsys Trigno IMUs, one strapped to each shank with Coban wrap. Each has an
accelerometer and a gyroscope. Both stream into one file per trial at about
148 Hz. The files look like this:

```
sub-P091_ses-S001_task-Pre_run-001_motion.xdf
    |          |         |        |
 subject    session    phase    trial number
```

Three phases:

| `task-` | What it is |
| --- | --- |
| `Pre` | Baseline, before anything was optimized |
| `Default` | One optimization trial, one device setting |
| `Post` | After optimization, to see what carried over |

---

## 2. Run this first

Save it in the repo root as `first_look.py`, change the path, and run
`python first_look.py` from that directory. It has to be the repo root, or
Python will not find `hitlo`.

```python
import numpy as np
from hitlo.io import load_streams
from hitlo.detectors import detect
from hitlo.symmetry import (walking_window, trim_peaks,
                            compute_step_times, compute_symmetry_index)

XDF = '/path/to/sub-P091_ses-S001_task-Pre_run-001_motion.xdf'
cfg = {'Sensing': {'backend': 'trigno', 'detector': 'gyro'}}

left, right = load_streams(XDF, cfg)
if left is None or right is None:
    raise SystemExit('a shank stream is missing from this file')

# Find the stretch of the recording that is actually walking.
win = walking_window(left, right)
if win is None:
    t0 = min(left.timestamps[0], right.timestamps[0])
    t1 = max(left.timestamps[-1], right.timestamps[-1])
else:
    t0, t1 = win

# Heel strikes, then drop the first and last 3 seconds of them.
lt = trim_peaks(np.sort(detect(left, cfg).heel_strike_times), t0, t1, 3.0)
rt = trim_peaks(np.sort(detect(right, cfg).heel_strike_times), t0, t1, 3.0)

right_steps, left_steps = compute_step_times(lt, rt)
si, per_stride = compute_symmetry_index(right_steps, left_steps, signed=True)

print(f'{len(lt)} left strikes, {len(rt)} right strikes')
print(f'SI = {si:+.2f} %   (stride to stride sd {per_stride.std():.2f})')
```

On `sub-P091` `task-Pre` `run-001` that prints:

```
103 left strikes, 102 right strikes
SI = +0.97 %   (stride to stride sd 1.73)
```

That is the whole pipeline. Everything below explains what those six steps did.

---

## 3. What each step did

**Load.** One file holds both sensors in a single stream, and the side is
written into the channel labels. `load_streams` splits them for you. It returns
two objects, each with `accel`, `gyro`, `timestamps`, `actual_fs` and `side`.
Either one can come back as `None`, which is why the script checks.

**Find the walking.** A recording is not walking end to end. The subject stands
while the operator sets up, walks, then stops while the recorder is still
going. `walking_window` returns the longest continuous stretch of real walking,
as a pair of timestamps. It returns `None` when it cannot find one, so always
write that `if win is None` fallback.

**Detect.** `detect` finds one heel strike per stride in each leg's gyroscope
signal. The rule: the shank swings forward fast, which makes a big peak, then
reverses when the foot lands. Contact is the moment the signal crosses zero
going downward, just after that peak. `docs/gyro_detection.md` has the details.

Two things `detect` works out on its own, because the sensors are not mounted
in a controlled orientation. First, which of the three gyro axes is the one the
leg rotates about. It picks the axis with the most variation. Second, which
direction is forward, since that depends on which way round the sensor was
clipped on. It takes the bigger swing to be forward. You will see both reported
in the QC plot as an axis letter and a "dominance" number.

**Trim.** `trim_peaks` drops heel strikes in the first and last 3 seconds.
Starting and stopping do not look like steady walking, and leaving them in
biases the result.

**Steps.** A step is one foot landing to the other foot landing.

| Term | Definition |
| --- | --- |
| Right step | Left foot lands, then right foot lands |
| Left step | Right foot lands, then left foot lands |
| Stride | Two steps, so one full cycle |

`compute_step_times` returns right steps first, then left. Getting that
backwards flips the sign of everything after it.

**Symmetry index.**

```
SI = 2 × (right step − left step) / (right step + left step) × 100 %
```

| SI | Meaning |
| --- | --- |
| 0 | Both steps take the same time |
| Above 0 | Right step is longer |
| Below 0 | Left step is longer |
| About ±1 to ±2 | Normal for a healthy walker |

---

## 4. Always look at the plot

```python
from hitlo.plot_heelstrikes import analyze_gyro, make_gyro_plot

qc = analyze_gyro(XDF, 3.0)       # does everything in section 2, in one call
make_gyro_plot(qc)                # or make_gyro_plot(qc, save_path='qc.png')
```

Or from the terminal, without writing a script:

```
python hitlo/plot_heelstrikes.py <file.xdf> --detector gyro --trim 3.0
```

The `--detector gyro` flag is required. The script still defaults to the old
accelerometer view.

Three panels: each leg's signal with the detected strikes marked, then the step
times underneath.

**Do this for every trial, not a sample of them.** A symmetry index is one
number summarizing a thousand events, and it will report a perfectly plausible
value for a detection that is completely wrong. The plot is where wrong
detection is obvious. It takes five seconds.

---

## 5. Is this trial any good

| Check | Want | If not |
| --- | --- | --- |
| Strike counts | Left and right within one or two | One leg is finding events the other is not |
| Stride times | Both legs similar, usually 1.0 to 1.5 s | If one is half the other, that leg is being counted twice |
| Stride to stride sd | A few points | A big number means the average is not describing steady walking |
| Dominance | Above 1.6 on both legs | Below that, the axis choice is close to a coin flip |
| The two traces | Similar shape | If one looks upside down, the direction was picked wrong on that leg |

The example trial passes all five: 103 against 102 strikes, 1.117 s strides,
sd 1.73, dominance 2.98 and 2.63.

There is an automatic version of the first three:

```python
from hitlo.symmetry import leg_consistency
for warning in leg_consistency(lt, rt, per_stride):
    print(warning)
```

It prints warnings and changes nothing. Reading them is your job.

---

## 6. Exercises

### A. Compare five trials

Run `analyze_gyro` on five trials from one subject and fill this in. Use
`qc['si']`, `qc['n_left']`, `qc['n_right']`, `np.std(qc['per'])`, and
`qc['left']['dominance']`.

| trial | SI | left | right | sd | dominance L / R |
| --- | --- | --- | --- | --- | --- |
| Pre run-001 | | | | | |
| Pre run-002 | | | | | |
| Default run-005 | | | | | |
| Default run-010 | | | | | |
| Post run-003 | | | | | |

Then answer:

1. Which trial is the most symmetric, and which is the least?
2. Between Pre run-001 and Pre run-002 the subject put the exoskeleton on, with
   the motor switched off. How much did SI change? Is that bigger or smaller
   than the differences between the Default trials?
3. Does any trial fail a check from section 5?

For `sub-P091` the answers are at the bottom of this page. Do not look first.

### B. Check the formula on one stride

The pipeline gives you per-stride values, but it is worth confirming the
formula does what you think once.

```python
qc = analyze_gyro(XDF, 3.0)
r = qc['r_steps'][0]      # first right step, in seconds
l = qc['l_steps'][0]      # first left step
print(r, l, qc['per'][0])
```

Work out `2 × (r − l) / (r + l) × 100` on a calculator and check it against
`qc['per'][0]`. On `sub-P091` `Pre` `run-001` the first stride is a right step
of 0.5462 s and a left step of 0.5527 s, which gives −1.17.

Notice how small the difference is. Six milliseconds out of half a second
becomes more than a point of SI. That is why the detector interpolates contact
times between samples instead of rounding to the nearest one.

### C. See how much the trim matters

Run `analyze_gyro` on the same file three times with `trim` of 0, 3 and 10
seconds. Note the SI and the strike count each time. How much does throwing
away 18 strides move the answer?

---

## 7. Two things that will bite you

**Forgetting `cfg`.** `load_streams(path)` without the config dictionary
silently uses the old Polar path and returns `(None, None)`. Define `cfg` once
at the top and pass it to both `load_streams` and `detect`.

**Mixing detectors.** The gyroscope marks the moment of contact. The
accelerometer marks the impact shock that follows it, tens of milliseconds
later. Numbers from the two are not comparable, so never check a gyro result
against a baseline that was measured with the accelerometer.

---

## Where to go next

| Document | Covers |
| --- | --- |
| `docs/gyro_detection.md` | The detection rule in full. Read this next. |
| `docs/trigno_setup.md` | The hardware and how a session is recorded |
| `docs/workflow.md` | How an experiment day runs |
| `docs/detection_pipeline.md` | The old accelerometer method. Background only. |

---

## Answers to exercise A

For `sub-P091` `ses-S001`:

| trial | SI | left | right | sd | dominance L / R |
| --- | --- | --- | --- | --- | --- |
| Pre run-001 | +0.97 | 103 | 102 | 1.73 | 2.98 / 2.63 |
| Pre run-002 | −9.07 | 96 | 96 | 2.32 | 3.53 / 2.91 |
| Default run-005 | −6.44 | 42 | 42 | 2.56 | 3.32 / 2.99 |
| Default run-010 | −11.04 | 37 | 37 | 2.77 | 2.83 / 2.78 |
| Post run-003 | −3.69 | 138 | 137 | 2.07 | 3.07 / 2.71 |

1. Pre run-001 is the most symmetric at +0.97. Default run-010 is the least at
   −11.04.
2. SI moved 10 points, from +0.97 to −9.07, just from wearing the device with
   the motor off. The Default trials sit between −6.44 and −11.04, a spread of
   about 5 points. So putting the device on did roughly twice as much as
   anything the optimizer did afterward. That is worth knowing before you
   interpret any of the Default trials.
3. No. All five pass. Note that the Default trials have far fewer strikes,
   around 40 against 100, because those trials are shorter.
