# Getting Started with HITLO_Symmetry

A 10-minute guide to going from "I just cloned this repo" to "I can run an experiment."

If something doesn't work, see the [Troubleshooting](#troubleshooting) section at the bottom.

---

## What this code does (in one paragraph)

HITLO_Symmetry runs a Bayesian-optimization-driven experiment to tune a passive
ankle exoskeleton. Per trial: the code suggests one **stiffness index** value
`x ∈ [−1, +1]`, which resolves through a pre-validated table into the four
settings the experimenter dials in (R, theta, L₀, attachment ratio). The
participant walks while two shank-mounted IMUs stream over LSL, the code
computes a step-time symmetry index from the recording, and feeds that back to
the optimizer. The first 5 trials follow a fixed ramp covering both the
dorsiflexor and plantarflexor arms; the optimizer runs after.

---

## What you need

**Hardware** — one of two sensor setups:
- LegExoNET passive ankle exoskeleton (or compatible spring-pulley device)
- **Trigno**: 2× Delsys Trigno Avanti on the shanks (accel + gyro, ~148 Hz),
  plus a machine running the Trigno Control Utility and an LSL bridge
- **Polar**: 2× Polar H10 chest straps worn on the shanks (accel only,
  ~200 Hz), with Coban wrap for skin contact
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
- `Subject.base_dir`: where data is saved (default `~/HITLO_Data`)
- `Sensing.backend`: `trigno` or `polar` — this decides everything downstream,
  including which detector runs and whether trials land in `motion/` or `eeg/`
- `Sensing.detector`: `gyro` or `accel`. Leave it out and each backend uses its
  default (gyro for Trigno, accel for Polar — Polar has no gyroscope).
- `Sensing.stream`: Trigno only, the LSL stream name your bridge publishes

**Polar only**: copy `config/sensors.example.json` to `config/sensors.json` and
fill in your two BLE device IDs. If you do not know them, run
`python apps/collect_sensors.py left` once — it scans and prints every
discoverable Polar in range.

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

Mount one sensor on each shank, medial side, just above the ankle at the bottom
of the muscle belly. Polar straps need Coban wrap for skin contact; Trigno
sensors clip on.

### Step 2: Bring up the streams

```bash
streamlit run apps/hitlo_console.py
```

Open the **Sensors** page. It adapts to your backend:

- **Trigno** — pairing and slot assignment happen in the Trigno Control Utility
  on the base-station machine, so there is nothing here to scan or start. The
  page *verifies*: stream present, channel count, rate, host, and which body
  segments arrived. You want `left_shank` and `right_shank` both listed.
- **Polar** — scan, assign sides, and start one process per sensor.

Either way, **run the shake test before trusting the sides.** Sensors get
swapped between sessions, and a swap inverts the sign of the symmetry index
while producing perfectly plausible numbers. `./apps/verify_sides.py` does the
same check from the terminal.

If a Polar sensor fails to connect, see [Troubleshooting](#sensor-wont-connect).

### Step 3: Configure LabRecorder

1. Open LabRecorder
2. Click **Update** → your streams appear (`TrignoIMU`, or `polar accel left`
   and `polar accel right`)
3. Check the boxes
4. **Save directory** — must match your config:
   `~/HITLO_Data/sub-P000/ses-S001/motion/` for Trigno,
   `.../eeg/` for Polar
5. **Filename template:** `sub-P000_ses-S001_task-Default_run-%n_motion.xdf`
   (`_eeg.xdf` for Polar; `%n` auto-increments)

### Step 4: Start the experiment UI

The console is already running from Step 2. In the sidebar, click
**🔄 Initialize/Reset System**.

You should see in Terminal 3:
```
✅ SymmetryCost v2.0.0 initialized
✅ HIL_Exo initialized
###### Generating 5 LHS exploration parameters ######
   ── Trial 1/5 ──
   ✅ LHS  R=0.XXXX  L0=0.XXXX | PF=0.00 Nm  DF=X.XX Nm  ...
```

The browser shows "Trial 1/15" with the index value and the device settings it
resolves to.

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
- `~/HITLO_Data/sub-P000/ses-S001/{motion,eeg}/gait_asymmetry_timeline.png`
- `~/HITLO_Data/sub-P000/ses-S001/derivatives/hil_optimization/visualizations/`

The console's **Results** page shows the GP posterior and the configuration to
carry forward. For a full post-hoc pass over a session:

```bash
python scripts/analyze_experiment.py --base-dir ~/HITLO_Data
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
| `tests/` | Regression suite — `./tests/test_regression.py` |

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

- **Trigno**: the bridge runs on the base-station machine, not this one. Check
  it is publishing, and that this machine can see it on the network — the
  console does not filter by host.
- **Polar**: make sure the H10 is on (button battery installed, snapped to the
  strap)
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
streamlit run apps/hitlo_console.py
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
