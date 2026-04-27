# HITLO_Symmetry — Gait Symmetry Cost for HITLO

Symmetry-based cost function for human-in-the-loop Bayesian optimization of a
passive ankle exoskeleton, built on top of Dr. Myunghee Kim's HIL_toolkit.
Optimizes spring parameters (anchor position **R**, rest length **L₀**) using
shank-mounted IMU gait symmetry as the cost signal — developed for post-stroke
gait rehabilitation.

**Platform:** LegExoNET passive ankle exoskeleton
**Sensors:** Two Polar H10 accelerometers, shank-mounted bilaterally

---

## Dependencies

This project builds on **[HIL_toolkit](https://github.com/UICRRL/HIL_toolkit)**
by Dr. Myunghee Kim's lab (UIC Rehab Robotics Lab), which provides the
Bayesian optimization engine (GP regression + expected improvement
acquisition) and Polar H10 BLE streaming utilities.

**HITLO_Symmetry contributes on top of that foundation:**
- Custom cost function based on step-time symmetry (not metabolic cost)
- Two-sensor shank-mounted IMU heel-strike detection pipeline
- Streamlit experimenter UI
- Exoskeleton-specific spring penalty adn safety constraints

---

## Installation

```bash
# 1. Install HIL_toolkit first (external dependency)
git clone https://github.com/UICRRL/HIL_toolkit.git
cd HIL_toolkit
pip install -e .
cd ..

# 2. Clone and install HITLO_Symmetry
git clone https://github.com/<your-username>/HITLO_Symmetry.git
cd HITLO_Symmetry
pip install -r requirements.txt
```

See [docs/getting_started.md](docs/getting_started.md) for a 10-minute setup
walkthrough including hardware setup and troubleshooting.

---

## Quick start

```bash
# 1. Connect the two Polar H10 sensors (in separate terminals)
python apps/collect_sensors.py right
python apps/collect_sensors.py left

# 2. Open LabRecorder, confirm both streams visible, click Update

# 3. Start the experiment UI
streamlit run apps/run_experiment.py
```

See [docs/workflow.md](docs/workflow.md) for the full experiment-day procedure.

---

## Project structure

```
HITLO_Symmetry/
├── hitlo/                    # core library (import this)
│   ├── detection.py          # heel-strike detection pipeline
│   ├── symmetry.py           # step-time interleaving + SI computation
│   ├── cost.py               # BO cost function (SymmetryCost class)
│   ├── io.py                 # XDF loading, trial-file naming
│   └── hil_exo.py            # HIL_Exo experiment driver (wraps HIL_toolkit's BO)
│
├── apps/                     # user-facing tools
│   ├── run_experiment.py     # Streamlit UI for live BO trials
│   ├── diagnose_trial.py     # standalone trial QC plotter
│   ├── gp_viewer.py          # interactive GP surface viewer (post-hoc)
│   └── collect_sensors.py    # BLE sensor startup script
│
├── scripts/                  # batch / dev utilities
│   └── analyze_experiment.py # full post-hoc session analysis
│
├── config/
│   └── exo_symmetry_config.example.yml  # template — copy to exo_symmetry_config.yml
│
├── docs/                     # extended documentation
│   ├── getting_started.md    # 10-minute setup + first run guide
│   ├── workflow.md           # experiment-day procedure
│   ├── detection_pipeline.md # algorithm details + references
│   └── porting_to_other_devices.md  # adapting to non-LegExoNET devices
│
└── tests/                    # unit tests (currently minimal)
```

---

## What this code does

Each trial, the BO loop (from HIL_toolkit) suggests a new `(R, L₀)` pair. The
experimenter physically adjusts the exoskeleton to those values, and the
participant walks for 60 seconds while two shank-mounted Polar H10 sensors
stream acceleration over Bluetooth (via LSL). After the trial,
`hitlo.cost.SymmetryCost.extract_cost_from_file()`:

1. Loads the XDF recording (via `hitlo.io`)
2. Runs the detection pipeline on each shank signal (`hitlo.detection`)
3. Interleaves left/right heel strikes → step times → symmetry index (`hitlo.symmetry`)
4. Adds a spring-shape penalty (prefers dorsiflexion-assist spring profiles) THESE VALUES ARE SET TO ZERO FOR ME!
5. Returns the total cost

The GP-based BO picks the next suggestion to minimize this cost.

---

## Detection pipeline (one-paragraph summary)

Raw tri-axial acceleration → magnitude `|a|` → jerk `|d|a|/dt|` → 15 Hz
lowpass Butterworth → z-scored. Candidate peaks detected with a strict
threshold (0.7 SD) and a gap-fill recovery pass (1.8 SD in anomalously long
gaps). Candidates grouped into gait-cycle clusters (peaks within 0.65 s).
Within each cluster, scan from the last peak backwards; pick the first one
that is (a) above gravity baseline (not a free-fall trough) AND (b) followed
by a stance region (flat signal near baseline). That's the heel strike. Each
cluster emits exactly one heel strike (or zero if nothing qualifies). Edge
singletons dropped; trial ends trimmed 3 seconds each way.

Full methodology, physiologic justification, and literature references in
[docs/detection_pipeline.md](docs/detection_pipeline.md).

If you want to use this codebase for a **different exoskeleton or robotic
device** (different mechanism, different parameters), see
[docs/porting_to_other_devices.md](docs/porting_to_other_devices.md) for a
walkthrough of what to change.

---

## Versioning

- **v2.0.0** (current, this repo) — refactored into library structure; detection
  logic consolidated into `hitlo.detection` as single source of truth
- **v1.8.0** — flat-file layout; cluster-keep-last added to `symmetry_cost.py`
- **v1.7.2** — LSL timestamp fix (two-sensor drift)
- **v1.6.0** — jerk-based detection replaces magnitude-peak detection

---

## Hardware

- LegExoNET passive ankle exoskeleton (spring-pulley mechanism)
- 2× Polar H10 chest straps → worn on shanks with Coban wrap
- Mac laptop with LabRecorder, LSL, Python 3.9+

Sensor IDs (THESE WILL CHANGE DEPEDNING ON SENSOR USED:
- Left shank: `7F302C25`
- Right shank: `80AE3629`

---

## Authors

- **Mac Camardo** — PhD Student, UIC Biomedical Engineering /
  Shirley Ryan AbilityLab. [marcc2@uic.edu](mailto:marcc2@uic.edu)
- **Dr. James Patton** (sponsor) — UIC BME / Shirley Ryan AbilityLab
- **Dr. Myunghee Kim** (co-sponsor) — UIC BME, author of HIL_toolkit

---

## Citation

If you use this code, please cite both the HITLO_Symmetry method paper (in prep)
and the underlying HIL_toolkit
---

## License

MIT (see [LICENSE](LICENSE))

HIL_toolkit is separately licensed; see its
[repository](https://github.com/UICRRL/HIL_toolkit) for details.

---

## Research use disclaimer

This is **research code** for an investigational device used under IRB-approved
human subjects protocols. It is **not** an FDA-cleared medical device. The
exoskeleton's safety constraints (90 Nm hard cap, PF zone limits, top-K
acquisition fallback) protect against most failure modes we've encountered, but
this code should only be used by trained researchers in a supervised lab
setting with appropriate participant safety procedures (gait belt, treadmill
emergency stop, screening for contraindications).

If you are adapting this for your own work and have questions about safety or
clinical use, contact Mac Camardo before deploying.
