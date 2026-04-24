# Experiment workflow

This is the step-by-step procedure for running a HITLO session.

## Before the participant arrives

1. **Charge both Polar H10 sensors** overnight.
2. **Apply the Coban wrap** to the right sensor (wraps around the shank for reliable skin contact). The right sensor has less surface area against the skin and needs this.
3. **Verify sensor IDs in config**:
   ```yaml
   # config/exo_symmetry_config.yml
   sensors:
     left_id:  "7F302C25"
     right_id: "80AE3629"
   ```
4. **Open 3 terminal windows**.

## Sensor startup

In order — left before right OR right before left, but each fully connected before starting the next.

**Terminal 1:**
```bash
cd ~/HITLO
python apps/collect_sensors.py right
```
Wait for the `connected` message.

**Terminal 2:**
```bash
python apps/collect_sensors.py left
```
Wait for `connected`.

**LabRecorder:**
1. Open LabRecorder
2. Click **Update** — you should see `polar accel left` and `polar accel right`
3. Check both boxes
4. Set save directory to `data/sub-<ID>/ses-<SESSION>/eeg/`
5. Filename template: `sub-<ID>_ses-<SESSION>_task-Default_run-<TRIAL>_eeg.xdf`

## Participant setup

1. Attach both Polar H10 sensors to the shanks (bottom of the muscle belly, just above the medial malleolus).
2. The right sensor goes inside the Coban wrap.
3. Attach the LegExoNET exoskeleton.
4. Verify in the experiment UI that both sensors are streaming (live plots visible).

## Running trials

**Terminal 3:**
```bash
streamlit run apps/run_experiment.py
```

For each trial:

1. UI shows the next `(R, L₀)` to set. **Read these carefully** — they're in meters.
2. Physically adjust the exoskeleton to those values.
3. In LabRecorder, click **Start**.
4. Participant walks for the trial duration (60 s default).
5. Click **Stop** in LabRecorder.
6. Click **Analyze Trial** in the UI.
7. Review the QC plot. If warnings appear (red 🚨 banners), pause and investigate before accepting the trial.
8. The BO will automatically suggest the next `(R, L₀)`.

## Between trials

Give the participant 30-60 seconds of rest. For older or more fatigued participants, allow longer.

## End of session

1. Stop both sensor terminals (Ctrl+C).
2. The UI auto-saves a checkpoint — you can close Streamlit.
3. Raw XDFs are in `data/sub-<ID>/ses-<SESSION>/eeg/`.
4. BO state + results summary in `data/sub-<ID>/ses-<SESSION>/derivatives/hil_optimization/`.

## If something goes wrong mid-session

- **Sensor disconnects:** UI shows a red "SENSOR DISCONNECTED" banner.
  Stop the affected `collect_sensors.py` process, restart it, click the
  Reconnect button in the UI.
- **Bad trial** (QC warnings, subject stumbled, etc.): delete the XDF,
  decrement the UI trial counter via the sidebar, redo the trial.
- **Streamlit crash:** reopen with the same command; the UI auto-resumes
  from checkpoint.

## Post-session analysis

Run the full-trial diagnostic on each recording:

```bash
python apps/diagnose_trial.py data/sub-P048/ses-S001/eeg/sub-P048_ses-S001_task-Default_run-001_eeg.xdf
```

This produces the 4-panel diagnostic figure showing detection quality, with
sanity-check warnings printed to the terminal.

For a batch summary across all trials:

```bash
python scripts/batch_analyze.py data/sub-P048/ses-S001/eeg/
```
