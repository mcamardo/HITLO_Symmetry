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

Per-trial detection quality:

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
