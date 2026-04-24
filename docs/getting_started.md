# Getting Started with HITLO_Symmetry

A 10-minute guide to going from "I just cloned this repo" to "I can run an experiment."

If something doesn't work, see the [Troubleshooting](#troubleshooting) section at the bottom.

---

## What this code does (in one paragraph)

HITLO_Symmetry runs a Bayesian-optimization-driven experiment to tune a passive ankle exoskeleton. Per trial: the code suggests spring parameters (anchor position **R**, rest length **L₀**), the experimenter physically sets the exo to those values, the participant walks for 60 seconds while two shank-mounted Polar H10 IMUs stream acceleration over Bluetooth, the code computes a step-time symmetry index from the recording, and feeds that back to the BO. After 5 exploration trials and 10 BO trials, you get the optimized spring parameters for that participant.

---

## What you need

**Hardware:**
- LegExoNET passive ankle exoskeleton (or compatible spring-pulley device)
- 2× Polar H10 chest straps — one mounted on each shank
- Coban wrap (ACE bandage) for sensor mounting
- Mac laptop with Python 3.9+

**Software:**
- This repo (HITLO_Symmetry)
- HIL_toolkit (Dr. Myunghee Kim's BO library — separate install)
- LabRecorder
- LSL libraries

---

## Setup (one-time, ~30 minutes)

### 1. Install HIL_toolkit (Bayesian optimization engine)

```bash
git clone https://github.com/UICRRL/HIL_toolkit.git
cd HIL_toolkit
pip install -e .
cd ..
```

### 2. Install HITLO_Symmetry

```bash
git clone git@github.com:mcamardo/HITLO_Symmetry.git
cd HITLO_Symmetry
pip install -r requirements.txt
```

### 3. Set up your config

```bash
cp config/exo_symmetry_config.example.yml config/exo_symmetry_config.yml
```

Now edit `config/exo_symmetry_config.yml` in any text editor:

- `Subject.id`: change to your participant ID (e.g. `P049`)
- `Subject.session`: usually `S001` for first session
- `Subject.base_dir`: change to where you want data saved (e.g. `/Users/yourname/HITLO`)
- `sensors.left_id` and `sensors.right_id`: the MAC suffixes of your Polar H10s

If you don't know your sensor IDs, run `python apps/collect_sensors.py left` once — it'll scan and print all discoverable Polars in the area.

### 4. Install LabRecorder

Download from <https://github.com/labstreaminglayer/App-LabRecorder/releases>. Drag the `.app` to `/Applications/`. Run once to confirm it opens.

### 5. Verify your install works

```bash
python -c "from hitlo.detection import detect_heelstrikes_full; print('OK')"
```

If that prints `OK`, you're set. If you see `ImportError`, jump to [Troubleshooting](#troubleshooting).

---

## Your first practice run (15 minutes)

Strongly recommended: **do this once before running on a real participant**, with yourself or a labmate as the test subject. It catches any setup issues without wasting a participant's time.

### Step 1: Wear the sensors

- Mount left Polar H10 on the left shank (medial side, just above the ankle)
- Mount right Polar H10 on the right shank, with Coban wrap for skin contact
- Make sure both are turned on (snap a button battery in if needed)

### Step 2: Start sensor streaming (3 terminals)

**Terminal 1:**
```bash
cd HITLO_Symmetry
python apps/collect_sensors.py right
```
Wait for `connected` message.

**Terminal 2:**
```bash
python apps/collect_sensors.py left
```
Wait for `connected`.

If a sensor fails to connect, see [Troubleshooting](#sensor-wont-connect).

### Step 3: Configure LabRecorder

1. Open LabRecorder
2. Click **Update** → both `polar accel left` and `polar accel right` should appear in the stream list
3. Check both boxes
4. **Save directory:** `~/HITLO/sub-P000/ses-S001/eeg/` (or whatever matches your config)
5. **Filename template:** `sub-P000_ses-S001_task-Default_run-%n_eeg.xdf` (the `%n` auto-increments)

### Step 4: Start the experiment UI

**Terminal 3:**
```bash
cd HITLO_Symmetry
streamlit run apps/run_experiment.py
```

A browser tab will open at `http://localhost:8501`. In the sidebar, click **🔄 Initialize/Reset System**.

You should see in Terminal 3:
```
✅ SymmetryCost v2.0.0 initialized
✅ HIL_Exo initialized
###### Generating 5 LHS exploration parameters ######
   ── Trial 1/5 ──
   ✅ LHS  R=0.XXXX  L0=0.XXXX | PF=0.00 Nm  DF=X.XX Nm  ...
```

The browser shows "Trial 1/15" with R and L₀ values.

### Step 5: Run a trial

For each trial:

1. **Read R and L₀** from the UI
2. **Set the exoskeleton** to those values
3. **Start LabRecorder** (the file auto-numbers to run-001, run-002, ...)
4. **Walk for 60 seconds** at a comfortable pace
5. **Stop LabRecorder** when timer ends
6. In the UI, click **Check File** → should turn green ✅
7. Click **Analyze Trial**
8. Review the heel strike QC plot that appears
9. UI auto-advances to the next trial

Repeat 15 times. The first 5 trials are LHS exploration; trials 6-15 use the BO suggestions.

### Step 6: Check the results

When all 15 trials are done, run:

```bash
python scripts/analyze_experiment.py --subject P000 --session S001
```

This generates timeline plots, BO iteration visualizations, and the all-trials torque curve grid in:
- `~/HITLO/sub-P000/ses-S001/eeg/gait_asymmetry_timeline.png`
- `~/HITLO/sub-P000/ses-S001/derivatives/hil_optimization/visualizations/`

You can also browse the GP iterations interactively:

```bash
streamlit run apps/gp_viewer.py -- --subject P000 --session S001
```

---

## What to expect during a session

| Trial range | What happens |
|---|---|
| 1–5 | LHS exploration — well-spread parameters across the search space |
| 6–15 | Bayesian optimization — GP picks the next best (R, L₀) to try |

**During BO, expect occasional safety messages like:**

```
⚠️  BO suggestion R=0.3500, L0=0.3000 failed (max=187.27 Nm)
🔍 Searching top-K acquisition rankings on grid...
✅ Top-K safe fallback (rank #14, EI=0.0287): R=0.3245, L0=0.3088
```

This is normal and expected. The safety system caught a parameter combination that would have produced excessive torque, and substituted the highest-EI safe point. The trial proceeds with the safe replacement.

---

## Repository tour

| Folder | What's there |
|---|---|
| `hitlo/` | Core library — detection, symmetry, cost, BO wrapper |
| `apps/` | Things you run during/after experiments (Streamlit UIs, CLI tools) |
| `scripts/` | Batch utilities (e.g. `analyze_experiment.py`) |
| `config/` | YAML config files |
| `docs/` | Detailed documentation (workflow, detection algorithm) |
| `tests/` | Unit tests |

If you want to **change algorithm behavior**, edit something in `hitlo/`. If you want to **change experiment parameters**, edit `config/exo_symmetry_config.yml` (no code changes needed).

---

## Troubleshooting

### `ImportError: No module named hitlo`

You're running Python from outside the repo root, or you haven't `pip install`'d everything in `requirements.txt`. From the repo root:
```bash
pip install -r requirements.txt
python -c "import hitlo"
```

### `ImportError: No module named HIL.optimization.BO`

You skipped step 1 of setup. Install HIL_toolkit:
```bash
git clone https://github.com/UICRRL/HIL_toolkit.git
cd HIL_toolkit
pip install -e .
```

### Sensor won't connect

- Make sure the Polar H10 is on (button battery installed, snapped to chest strap)
- Move the laptop closer (BLE has limited range)
- Restart Bluetooth: turn it off and back on in macOS settings
- Try the other sensor first to confirm one works before debugging both

If `apps/collect_sensors.py left` hangs at "Scanning for Polar H10 ..." for >30 seconds, the sensor isn't being seen. Take it off, wait 10 seconds, put it back on, and retry.

### Streamlit shows "File not found" after recording

Check that:
1. LabRecorder was actually clicked **Start** (not just running) — check it shows "Recording"
2. The save directory matches what's in your config (`{base_dir}/sub-{id}/ses-{session}/eeg/`)
3. The filename matches `sub-{id}_ses-{session}_task-Default_run-%n_eeg.xdf` exactly

### "QC failed: not enough heel strikes"

Either:
- The participant didn't walk long enough (need at least ~10 strides after trim)
- The sensor lost contact mid-trial (check the live plot in the UI for dropouts)
- The detection threshold is too strict for this participant (rare; for very mild gait, try lowering `strict_thresh` in `hitlo/detection.py`)

Just redo the trial. Delete the bad XDF, click **Check File** again, re-record.

### Streamlit crashed mid-experiment

Just restart it:
```bash
streamlit run apps/run_experiment.py
```

It auto-resumes from the saved checkpoint. The trials you've already run are preserved.

---

## Where to learn more

- **`docs/workflow.md`** — full experiment-day procedure with timing
- **`docs/detection_pipeline.md`** — the heel-strike detection algorithm explained, with literature references
- **`README.md`** — high-level overview, citations, project structure

---

## Who to ask

- **Algorithm questions** (detection, BO, cost function): Mac Camardo
- **Hardware questions** (exoskeleton, sensors): Mac or Dr. Patton
- **HIL_toolkit / GP regression**: Dr. Myunghee Kim
- **General lab help**: anyone in Patton or Kim labs at UIC/SRAL

---

*Last updated: April 2026 — HITLO_Symmetry v2.3.0*
