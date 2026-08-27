# Experiment workflow

Step-by-step procedure for running a HITLO session. Everything routes through
the console — `streamlit run apps/hitlo_console.py` — which adapts to the
sensor backend set in `config/exo_symmetry_config.yml`.

```yaml
Sensing:
  backend: trigno      # or: polar
  detector: gyro       # or: accel
  stream: TrignoIMU    # trigno only — the LSL stream the bridge publishes
```

The backend also decides where trials are written: `motion/` for Trigno,
`eeg/` for Polar (an artefact of the original LabRecorder template, kept so
existing recordings still load).

---

## Before the participant arrives

**Both backends**

1. Confirm `Subject.id` and `Subject.session` in the config. Using a subject ID
   that already has recordings will interleave new trials with old ones.
2. Run the readiness check:
   ```bash
   ./apps/preflight.py
   ```
   It verifies the config, the data directory, the index table, LSL streams and
   the most recent recording. Fix anything it flags before the participant is on
   the treadmill.

**Trigno**

3. Power the sensors and confirm them in the Trigno Control Utility on the
   base-station machine, then start the bridge that publishes the LSL stream.
4. In the console's **Sensors** page, run *Look for the stream*. You want the
   expected channel count, a rate near 148 Hz, and both `left_shank` and
   `right_shank` listed in the inventory. Extra segments (foot, thigh) are
   carried but unused by the cost function.

**Polar**

3. Charge both H10 straps overnight; apply Coban wrap for skin contact.
4. In the console's **Sensors** page, scan, assign sides, and start one process
   per sensor.

---

## Participant setup

1. Mount the shank sensors just above the medial malleolus, at the bottom of
   the muscle belly, one per leg.
2. Attach the LegExoNET exoskeleton.
3. **Confirm which stream is which leg.** Sensors get swapped between sessions,
   and a swap inverts the sign of the symmetry index while producing entirely
   plausible numbers. Use the console's shake test, or:
   ```bash
   ./apps/verify_sides.py
   ```
4. Watch the live plots for a few strides before starting.

### Ankle calibration (only if you are recording ankle angle)

Step time and the symmetry index need none of this — skip the whole section if
you are not using a foot sensor.

There are **two** calibrations, and they do different jobs. The axis
calibration sets how large the angle is; the neutral pose sets where its zero
sits. Without the first, the shape of the curve is still readable but the
numbers are not: measured on P017, skipping it put range of motion near 140°
against a literature 25–30°.

#### 1. Axis calibration — three movements, once per session

Record these **as their own file**, before the walking trials, with the sensors
already mounted where they will stay. Do not fold them into a walking trial: a
stretch of walking can be mistaken for the third movement.

> **A.** Sit with the foot off the ground and the lower leg still. Starting
> with the foot relaxed, pull the toes up toward the shin and lower them again,
> about ten times. Only the foot moves.
>
> **B.** Stand with the foot flat and press the heel into the floor. Keeping
> the heel down, rock the knee forward over the toes and back, about ten times.
> Only the shank moves.
>
> **C.** Sit with the foot off the ground and hold the ankle stiff, as if it
> were in a walking boot. Swing the whole lower leg from the knee about ten
> times, so the foot and shank move together as one piece.

Each movement does one job. **A** measures the ankle axis as the foot's sensor
sees it, **B** measures the same axis as the shank's sensor sees it, and **C**
settles the sign between them — which is the part that actually broke: with the
relative sign wrong, the two segments' swings add instead of cancelling, and
that is where 140° came from.

The order does not matter and the parts do not need marking; they are
identified automatically by which segment is moving. `trial_explorer` reports
whether each movement isolated one axis, whether A and B found the same axis,
and whether the ankle really stayed locked during C, and refuses the
calibration rather than producing a plausible wrong magnitude.

Recovered a known ankle excursion to within 0.03° in simulation across four
sensor mountings and three ranges of motion. It is mounting-independent by
construction — strap the sensors on at any angle — but it does **not** survive
re-strapping, since the measured axis encodes where each sensor sits.

> The published alternative, a hinge fit over the whole recording (Seel,
> Raisch & Schauer 2014, *Sensors* 14(4) 6891–6909), is deliberately not used
> here. It assumes the ankle is a hinge; with 31–52% of the measured rotation
> off-axis in this data, its recovered axis lands ~39° from truth and range of
> motion swings from 27° on one leg to 121° on another.

#### 2. Neutral pose — sets the zero

Ankle angle also needs a neutral reference to be zeroed against. At the start
of the **first** trial after mounting, have the participant hold:

> **Stand tall and still, weight even on both feet, feet flat and pointing
> forward, knees straight but not locked, arms at your sides. Look ahead, not
> down. Hold for 10 seconds.**

Being specific matters. "Stand still" is not enough: a subject resting weight
on one leg, or with a knee soft, still produces a clean-looking zero that does
not correspond to the neutral their gait is measured against.

**Capture it once per mounting, not once per trial.** Measured on a recording
with two quiet stands 105 s apart, reusing the earlier one costs a 0.9 degree
offset and leaves the stride-averaged shape unchanged (r = 0.9999); gyro bias
moved 0.09 deg/s on the foot and 0.01 on the shank over that interval. What it
does *not* survive is a sensor being re-strapped — the zero encodes where the
sensor sits on the limb, so re-mount means re-capture.

`apps/trial_explorer.py` shows the captured window and checks it: duration,
stillness, that each sensor reads ~1 g, and that the shank is near vertical.
It rejects a window that is not a usable neutral rather than quietly producing
a plausible wrong angle.

---

## Running trials

For each trial:

1. The console shows the next index value `x` and the four device settings it
   resolves to (R, theta, L₀, attachment ratio). **Set the exoskeleton to those
   values** — the table is pre-validated, so do not improvise between rows.
2. In LabRecorder: Block/Task = `Default`, Run = the trial number shown, then
   **Start**.
3. Participant walks for the configured duration (90 s default).
4. **Stop** in LabRecorder.
5. Click **Analyze Trial** in the console.
6. Review the QC output. Red banners mean investigate before accepting.
7. The optimizer suggests the next `x`.

The first `manual_ramp_trials` (default 5) come from `ramp_sequence` rather
than the optimizer, so the GP starts with coverage of both the dorsiflexor and
plantarflexor arms instead of clustering in one.

---

## End of session

1. Stop any sensor processes (Polar backend only).
2. The console auto-saves a checkpoint; closing Streamlit is safe.
3. Raw XDFs: `<base_dir>/sub-<ID>/ses-<SESSION>/{motion,eeg}/`
4. BO state and results: `<base_dir>/sub-<ID>/ses-<SESSION>/derivatives/hil_optimization/`

---

## If something goes wrong mid-session

- **Sensor disconnects.** The console shows a red banner. On Trigno, both sides
  share one inlet, so a drop takes out both — restart the bridge, then
  re-attach from the console. On Polar, restart the affected sensor process.
- **Bad trial** (QC warnings, a stumble, a pause): delete the XDF, decrement
  the trial counter in the sidebar, redo it.
- **Streamlit crash.** Reopen with the same command; it resumes from checkpoint.

---

## Post-session analysis

Browse any recording interactively — pick a file, see where the walking was,
where every heel strike landed on the raw signal, and how the two detectors
compare. Zoomable, and it switches backends and detectors from the sidebar:

```bash
streamlit run apps/trial_explorer.py
```

Per-trial detection quality, as a one-shot figure:

```bash
./apps/diagnose_trial.py <path-to-trial>.xdf
```

Compare the two detectors on the same recording — useful whenever a trial's
numbers look surprising, since the methods fail on different strides:

```bash
./apps/compare_detectors.py <path-to-trial>.xdf
```

Full session summary:

```bash
python scripts/analyze_experiment.py --base-dir ~/HITLO_Data
```

See [detection_pipeline.md](detection_pipeline.md) and
[gyro_detection.md](gyro_detection.md) for what the detectors actually do, and
the Validation status section of the [README](../README.md) for what has and
has not been verified.
