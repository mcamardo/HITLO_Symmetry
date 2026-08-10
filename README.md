# HITLO_Symmetry — Gait Symmetry Cost for HITLO

Symmetry-based cost function for human-in-the-loop Bayesian optimization of a
passive ankle exoskeleton, built on top of Dr. Myunghee Kim's HIL_toolkit.
Optimizes spring parameters (**L₀** rest length, **attach** foot attachment)
using shank-mounted IMU gait symmetry as the cost signal — developed for
post-stroke gait rehabilitation.

**Platform:** LegExoNET passive ankle exoskeleton (spring-pulley mechanism)  
**Sensors:** Two Polar H10 accelerometers, shank-mounted bilaterally  
**Parameterization:** (L₀, attach) with R = 0.28 m fixed

---

## Dependencies

This project builds on **[HIL_toolkit](https://github.com/UICRRL/HIL_toolkit)**
by Dr. Myunghee Kim's lab (UIC Rehab Robotics Lab), which provides the
Bayesian optimization engine (GP regression + expected improvement
acquisition) and Polar H10 BLE streaming utilities.

**HITLO_Symmetry contributes on top of that foundation:**
- Custom cost function based on step-time symmetry (not metabolic cost)
- Two-sensor shank-mounted IMU heel-strike detection pipeline
- Streamlit experimenter UI (baseline-relative targeting, bidirectional HILBO)
- Exoskeleton-specific spring penalty and safety constraints
- (L0, attach) parameterization for intuitive device tuning

---

## Installation

```bash
# 1. Install HIL_toolkit first (external dependency)
git clone https://github.com/UICRRL/HIL_toolkit.git
cd HIL_toolkit
pip install -e .
cd ..

# 2. Clone and install HITLO_Symmetry
git clone https://github.com/mcamardo/HITLO_Symmetry.git
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
streamlit run apps/hitlo_console.py
```

See [docs/workflow.md](docs/workflow.md) for the full experiment-day procedure.

---

## Project structure

```
HITLO_Symmetry/
├── hitlo/                         # core library (import this)
│   ├── detection.py               # heel-strike detection pipeline
│   ├── symmetry.py                # step-time interleaving + SI computation
│   ├── cost.py                    # BO cost function (SymmetryCost class)
│   ├── io.py                      # XDF loading, trial-file naming
│   └── hil_exo.py                 # HIL_Exo experiment driver (wraps HIL_toolkit's BO)
│
├── apps/                          # user-facing tools
│   ├── hitlo_console.py           # Streamlit UI for live BO trials (MAIN)
│   ├── diagnose_trial.py          # standalone trial QC plotter
│   └── collect_sensors.py         # BLE sensor startup script
│
├── scripts/                       # batch / dev utilities
│   └── analyze_experiment.py      # full post-hoc session analysis
│
├── config/
│   └── exo_symmetry_config.example.yml  # template — copy to exo_symmetry_config.yml
│
├── docs/                          # extended documentation
│   ├── getting_started.md         # 10-minute setup + first run guide
│   ├── workflow.md                # experiment-day procedure
│   ├── detection_pipeline.md      # algorithm details + references
│   └── porting_to_other_devices.md  # adapting to non-LegExoNET devices
│
└── tests/                         # unit tests (currently minimal)
```

---

## What this code does

Each trial, the BO loop (from HIL_toolkit) suggests a new `(L0, attach)` pair:

- **L₀** (0.32–0.44 m) — spring rest length; controls engagement timing
- **attach** (−0.2 to +1.0) — foot attachment ratio; controls torque direction
  - Negative → plantarflexor-assist (longer paretic step time)
  - Positive → dorsiflexor-assist (exaggerated deficit, error-augmentation)
- **R** (0.28 m, fixed) — anchor distance from ankle center; controls moment arm

The experimenter physically adjusts the exoskeleton to those values, and the
participant walks for 60 seconds while two shank-mounted Polar H10 sensors
stream acceleration over Bluetooth (via LSL). After the trial,
`hitlo.cost.SymmetryCost.extract_cost_from_file()`:

1. Loads the XDF recording (via `hitlo.io`)
2. Runs the detection pipeline on each shank signal (`hitlo.detection`)
3. Interleaves left/right heel strikes → step times → symmetry index (SI) (`hitlo.symmetry`)
4. Adds a spring-shape penalty (prefers physiologic torque profiles)
   weighted by `lambda_pf` (plantarflexor) and `mu_df` (dorsiflexor) in config
5. Returns the total signed cost (SI in %, positive = longer right step)

The GP-based BO picks the next suggestion to minimize distance to the target SI.

---

## Detection pipeline (one-paragraph summary)

Raw tri-axial acceleration → magnitude `|a| = sqrt(x² + y² + z²)` → 50 Hz
lowpass Butterworth (filtfilt, zero phase delay) → differentiate → z-score.
The 50 Hz cutoff sits well above the heel-strike impact band (5–30 Hz), so
the lowpass acts as light noise cleanup rather than reshaping the impact.
Candidate peaks detected on jerk z-score with a strict threshold (0.7 SD) and
a gap-fill recovery pass (1.8 SD in anomalously long gaps). Candidates
grouped into gait-cycle clusters (peaks within 0.5 s). Within each cluster,
scan from the last peak backwards; pick the first one that is (a) above
gravity baseline (not a free-fall trough) AND (b) followed by a stance region
(flat signal near baseline). That's the heel strike. Each cluster emits
exactly one heel strike (or zero if nothing qualifies). Edge singletons
dropped; trial ends trimmed 3 seconds each way.

Symmetry index follows the standard form `SI = 2 × (R - L) / (R + L) × 100%`
where `R, L` are mean step times. Can be signed (captures asymmetry direction)
or unsigned (captures magnitude only).

Full methodology, physiologic justification, and literature references in
[docs/detection_pipeline.md](docs/detection_pipeline.md).

If you want to use this codebase for a **different exoskeleton or robotic
device** (different mechanism, different parameters), see
[docs/porting_to_other_devices.md](docs/porting_to_other_devices.md) for a
walkthrough of what to change.

---

## Configuration

Copy `config/exo_symmetry_config.example.yml` to `exo_symmetry_config.yml` and edit:

```yaml
Subject:
  id: P001                                    # participant ID
  session: S001                               # session code
  base_dir: /path/to/HITLO_Data               # data root

Cost:
  aim: Aim 1                                  # "Aim 1" (healthy) or "Aim 2" (stroke)
  sample_rate: 200                            # Polar H10 sampling rate (Hz)
  time: 90                                    # trial duration (seconds)
  signed: true                                # use signed SI (required for both aims)
  si_target: -10.0                            # fallback target (baseline phase overrides)
  trim_seconds: 3.0                           # trim from each end (steady-state window)

Optimization:
  n_parms: 2                                  # always 2 (L0, attach)
  n_steps: 15                                 # total trials
  n_exploration: 5                            # LHS exploration trials
  range: [[0.32, -0.2], [0.44, 1.0]]        # [L0_min, attach_min], [L0_max, attach_max]
  
  # Safety constraints (hard limits, always enforced)
  max_pf_torque_nm: 90.0                      # max plantarflexor torque (Nm)
  pf_check_angle_range: [0.0, 30.0]          # angle window for PF check (deg)
  max_df_torque_nm: 10.0                      # max dorsiflexor torque (Nm)
  df_check_angle_range: [-30.0, 0.0]         # angle window for DF check (deg)
  slack_at_neutral_max_torque: 2.0           # max slack at neutral (Nm)
  df_check_angle_deg: -10.0                   # where to check DF torque (deg)
  
  # Penalty weights (cost function shaping)
  lambda_pf: 0.01                             # plantarflexor penalty
  mu_df: 0.005                                # dorsiflexor penalty
  
  # BO settings
  normalize: true                             # normalize design space
  device: cpu                                 # torch device (cpu or cuda)
  acquisition: ei                             # qNoisyExpectedImprovement
  kernel_function: se                         # Matern 2.5 or RBF
```

---

## Baseline-relative targeting (Aim 1, healthy subjects)

For healthy subjects, the goal is to induce a fixed *displacement* from the
subject's own baseline asymmetry, not a fixed absolute SI. The baseline phase:

1. **Pre run-001** — no-device familiarization walk. **Not analyzed.**
2. **Pre run-002** — THE baseline trial (band slack, no perturbation).
   Its signed SI alone defines `baseline_si`.
3. **Target is computed as:**
   ```
   si_target = baseline_si - displacement
   ```
   Always pushed more negative, matching the left-side device geometry.

4. The BO then optimizes to minimize `|SI - si_target|`.

This ensures the induced asymmetry *dose* is held constant across subjects
with different baseline asymmetries (Patton et al. 2006 error-augmentation
principle).

---

## Versioning

- **v2.2.0** (current) — (L0, attach) parameterization with R = 0.28 m fixed;
  bidirectional HILBO for stroke; baseline-relative targeting for healthy;
  hard torque constraints with safety fallback
- **v2.1.0** — switched to filter-then-diff at 50 Hz cutoff; tightened cluster-gap
- **v2.0.0** — refactored into library structure
- **v1.8.0** — cluster-keep-last added
- **v1.6.0** — jerk-based detection

---

## Hardware

- LegExoNET passive ankle exoskeleton (spring-pulley mechanism, ~$5k)
- 2× Polar H10 chest straps → worn on shanks with Coban wrap (~$180)
- Mac laptop with LabRecorder, LSL, Python 3.9+

Sensor IDs (customize for your hardware):
- Left shank: `7F302C25`
- Right shank: `80AE3629`

---

## Authors

- **Mac Camardo** — PhD Candidate, UIC Biomedical Engineering /
  Shirley Ryan AbilityLab. [marcc2@uic.edu](mailto:marcc2@uic.edu)
- **Dr. James Patton** (advisor) — UIC BME / Shirley Ryan AbilityLab
- **Dr. Myunghee Kim** (co-advisor) — UIC BME, author of HIL_toolkit

---

## Citation

If you use this code, please cite both this work and the underlying HIL_toolkit:

**HITLO_Symmetry (in prep):**
```
Camardo, M., Patton, J. L., & Kim, M. (2026). 
Human-in-the-loop Bayesian optimization for personalized ankle exoskeleton 
gait rehabilitation using bilateral symmetry. [Journal]. 
```

**HIL_toolkit (Kantharaju & Kim, 2023):**
```
Kantharaju, P., & Kim, M. (2023). 
HIL_toolkit: A modular Bayesian optimization framework for 
human-in-the-loop robotic device optimization. IEEE RA-L, 8(4), 9813–9820.
```

---

## License

MIT (see [LICENSE](LICENSE)).

HIL_toolkit is separately licensed; see its
[repository](https://github.com/UICRRL/HIL_toolkit) for details.

---

## Research use disclaimer

This is **research code** for an investigational device used under IRB-approved
human subjects protocols. It is **not** an FDA-cleared medical device. The
exoskeleton's safety constraints (90 Nm hard cap, PF/DF zone limits, top-K
acquisition fallback) protect against most failure modes we've encountered, but
this code should only be used by trained researchers in a supervised lab
setting with appropriate participant safety procedures (gait belt, treadmill
emergency stop, screening for contraindications).

If you are adapting this for your own work and have questions about safety or
clinical use, contact Mac Camardo before deploying.
