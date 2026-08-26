# HITLO_Symmetry — Gait Symmetry Cost for HITLO

Symmetry-based cost function for human-in-the-loop Bayesian optimization of a
passive ankle exoskeleton, built on top of Dr. Myunghee Kim's HIL_toolkit.
Optimizes a single **unified stiffness index** — one number spanning
dorsiflexor-assist through zero torque to plantarflexor-assist — using
shank-mounted IMU gait symmetry as the cost signal. Developed for post-stroke
gait rehabilitation.

**Platform:** LegExoNET passive ankle exoskeleton (spring-pulley mechanism)  
**Sensors:** Shank-mounted IMUs, bilateral — Delsys Trigno Avanti (accel +
gyro, ~148 Hz) or Polar H10 (accel only, ~200 Hz), selected by config  
**Search space:** one index `x ∈ [−1, +1]`, each value resolving through a
pre-validated table to the four settings the operator dials in

---

## Dependencies

This project builds on **[HIL_toolkit](https://github.com/UICRRL/HIL_toolkit)**
by Dr. Myunghee Kim's lab (UIC Rehab Robotics Lab), which provides the
Bayesian optimization engine (GP regression + expected improvement
acquisition) and Polar H10 BLE streaming utilities.

**HITLO_Symmetry contributes on top of that foundation:**
- Custom cost function based on step-time symmetry (not metabolic cost)
- Two-sensor shank-mounted IMU heel-strike detection, accelerometer or
  gyroscope, behind one interface so the optimizer is sensor-agnostic
- Streamlit experimenter UI (baseline-relative targeting, bidirectional HILBO)
- Exoskeleton-specific spring penalty and safety constraints
- A unified 1-D stiffness index as the BO axis, replacing a multi-parameter
  search with one physically ordered knob

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
# Everything runs from the console — sensor setup, checks, trials, results.
streamlit run apps/hitlo_console.py
```

The console's **Sensors** page adapts to the backend in your config:

- **trigno** — pairing happens in the Trigno Control Utility on the
  base-station machine; the page verifies the LSL stream, channel labels,
  rate and host, and runs a shake test to confirm the side mapping
- **polar** — scans for the two H10 straps over BLE, assigns sides and
  starts one process per sensor

Standalone tools, if you prefer the terminal:

```bash
./apps/dry_run.py               # practise a session with no hardware
streamlit run apps/trial_explorer.py   # browse recordings and detection
./apps/preflight.py             # pre-session readiness check
./apps/verify_sides.py          # confirm which stream is which leg
./apps/diagnose_trial.py FILE   # per-trial QC plots
./apps/compare_detectors.py FILE  # accel vs gyro on the same recording
python apps/collect_sensors.py right    # Polar backend only
```

See [docs/workflow.md](docs/workflow.md) for the full experiment-day procedure.

---

## Project structure

```
HITLO_Symmetry/
├── hitlo/                         # core library (import this)
│   ├── detection.py               # accelerometer detection (jerk peaks + clustering)
│   ├── detection_gyro.py          # gyroscope detection (swing peak -> zero crossing)
│   ├── detectors.py               # dispatcher — picks the method from config
│   ├── symmetry.py                # step-time interleaving + SI computation
│   ├── cost.py                    # BO cost function (SymmetryCost class)
│   ├── io.py                      # XDF/LSL loading for both backends, file naming
│   ├── index_unified.py           # unified stiffness index (the BO's x axis)
│   ├── ankle_angle.py             # ankle angle from foot + shank IMUs (offline)
│   ├── plot_heelstrikes.py        # shared detection plotting
│   └── hil_exo.py                 # HIL_Exo experiment driver (wraps HIL_toolkit's BO)
│
├── apps/                          # user-facing tools
│   ├── hitlo_console.py           # Streamlit console — the main interface
│   ├── preflight.py               # pre-session readiness check
│   ├── verify_sides.py            # confirm stream-to-leg mapping
│   ├── trial_explorer.py          # browse any recording, interactively
│   ├── dry_run.py                 # whole optimization, simulated, no hardware
│   ├── diagnose_trial.py          # standalone trial QC plotter
│   ├── compare_detectors.py       # accel vs gyro on one recording
│   ├── check_placement.py         # compare sensor mountings empirically
│   ├── session.py                 # session/checkpoint helpers
│   └── collect_sensors.py         # Polar BLE sensor startup script
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
│   ├── detection_pipeline.md      # accelerometer algorithm + references
│   ├── gyro_detection.md          # gyroscope algorithm + what a bridge must provide
│   ├── trigno_setup.md            # Delsys hardware setup, end to end
│   └── porting_to_other_devices.md  # adapting to non-LegExoNET devices
│
└── tests/                         # regression suite (./tests/test_regression.py)
```

---

## What this code does

Each trial, the BO loop (from HIL_toolkit) suggests one number, `x ∈ [−1, +1]`:

- `x = −1` — maximum dorsiflexor stiffness (limited by the band ceiling)
- `x =  0` — zero torque
- `x = +1` — maximum plantarflexor stiffness (limited by the torque cap)

Each `x` resolves through `config/index_unified.csv` into the four physical
values the operator sets on the device: anchor distance **R**, angle
**theta**, spring rest length **L₀**, and foot **attachment ratio**. Every row
of that table is a pre-validated survivor of a 4-D sweep against the sign,
torque-cap, engagement and edge-cliff filters, so safety lives in the table
rather than in a runtime check.

Two properties of the index are load-bearing and easy to get wrong:

- **Never interpolate between rows.** Adjacent rows can differ wildly in every
  parameter; averaging two produces a configuration that passed none of the
  filters. Always snap to a row.
- **`x` is not uniformly spaced.** The two arms carry different level counts,
  so dorsiflexor steps are wider than plantarflexor steps. Match on nearest
  value in the `x` column; never compute a row position arithmetically.

The experimenter physically adjusts the exoskeleton to those values, and the
participant walks for 60 seconds while two shank-mounted IMUs stream over LSL.
After the trial,
`hitlo.cost.SymmetryCost.extract_cost_from_file()`:

1. Loads the XDF recording (via `hitlo.io`)
2. Runs heel-strike detection on each shank signal (`hitlo.detectors`
   dispatches to the accelerometer or gyroscope method)
3. Interleaves left/right heel strikes → step times → symmetry index (SI) (`hitlo.symmetry`)
4. Adds a spring-shape penalty (prefers physiologic torque profiles)
   weighted by `lambda_pf` (plantarflexor) and `mu_df` (dorsiflexor) in config
5. Returns the total signed cost (SI in %, positive = longer right step)

The GP-based BO picks the next suggestion to minimize distance to the target SI.

---

## Detection pipelines

Two methods, one interface. `hitlo.detectors.detect()` picks between them from
the config, so the cost function and optimizer never know which ran.

**Accelerometer** (`hitlo/detection.py`) — raw tri-axial acceleration →
magnitude `|a| = sqrt(x² + y² + z²)` → 50 Hz lowpass Butterworth (filtfilt,
zero phase) → differentiate → z-score. The 50 Hz cutoff sits well above the
heel-strike impact band (5–30 Hz), so the lowpass is light noise cleanup
rather than reshaping the impact. Candidate peaks on jerk z-score at 1.3 SD,
grouped into gait-cycle clusters (peaks within 0.5 s). Within each cluster,
scan back from the last peak and take the first that is above gravity
baseline and followed by a stance region. Each cluster emits at most one heel
strike. Edge singletons dropped; trial ends trimmed 3 s each way.

**Gyroscope** (`hitlo/detection_gyro.py`) — sagittal shank angular velocity,
bandpassed, oriented so mid-swing is positive. Contact is the first
negative-going zero crossing after each mid-swing peak, refined by sub-sample
interpolation and confirmed by a reversal check. Peak spacing adapts to the
subject's own cadence, with the stride estimated by autocorrelation rather
than from detected events (deriving it from detections is circular).

**Which to prefer.** At 148 Hz the accelerometer's heel-strike impact spans
only ~2 samples and its peak amplitude varies ~25% stride to stride, so a soft
strike can fall below threshold and be missed. The gyroscope's mid-swing
feature spans ~39 samples with ~10% amplitude variation, so it is far better
resolved at this sample rate and misses fewer strides. The gyro path is the
default for Trigno; Polar hardware has no gyroscope and uses the
accelerometer.

**They are not interchangeable measurements.** The gyro marks initial contact;
the accelerometer marks the impact shock that follows it, roughly 40–110 ms
later. A symmetry index from one cannot be compared against a baseline
collected with the other — switching detector mid-study means re-baselining.
`apps/compare_detectors.py` quantifies the offset on your own recordings.

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

## Validation status

Stated plainly, because the distinction matters when reading any number this
code produces.

**There is no external reference in this dataset.** No force plate,
instrumented treadmill, motion capture, or footswitches. Every check below
compares one IMU-derived estimate against another. Agreement between them is
*consistency*, not *accuracy* — a bias shared by both methods is invisible to
all of it.

**Tested against a known manipulation.** The one piece of evidence here that
does not reduce to IMUs agreeing with each other: three trials where the
subject walked normally, then with a deliberate limp on the right, then on the
left. Which leg was impaired is not in doubt.

| detector | normal | limp RIGHT | limp LEFT | verdict |
|---|---|---|---|---|
| gyro (shipping) | −4.93% | −2.42% | −15.00% | both directions correct, 12.6 pt separation |
| accelerometer | −4.39% | −0.07% | −3.65% | **limp LEFT comes out the wrong sign** |

The limps were not equally hard — measured at the step-time level, the left
limp shifted gait 67 ms against the right limp's 19 ms. Normalising for that,
the gyro reports 0.13 SI points per ms for one and 0.15 for the other, so its
response is proportional and even-handed; the raw asymmetry in the table is the
manipulation, not the detector. The accelerometer reports a shift toward the
right leg when the left was the impaired one, which is consistent with it
dropping nine strides in that trial.

This validates **direction and sensitivity**. It says nothing about absolute
timing accuracy — for that you still need an external reference.

**What else has been checked** (sub-P012/ses-S001, 8 walking trials, Trigno):

- The two detectors are independent in sensor, algorithm and body segment, and
  agree on the symmetry index to a mean of 3.0 points across trials.
- Gyro polarity is verified on 16/16 legs. An earlier heuristic picked the
  wrong lobe on every one of them, placing contact ~480 ms early; see the
  regression test `test_gyro_polarity_prefers_the_larger_lobe`.
- Stride counts, stride-time regularity and physiological plausibility of step
  times: the six normal-cadence trials produce zero implausible step times on
  the gyro path.
- 16 regression tests, including detection on synthetic signals with known
  contact times, dtype invariance, and a cross-backend end-to-end run.

**What has NOT been established:**

- **Absolute timing accuracy.** The gyro marks contact 40–110 ms before the
  accelerometer's impact peak; which is closer to true initial contact is
  unknown without an external reference.
- **The left leg.** The optional foot sensor has only ever been mounted on the
  right, so the left — the exo side, where a damped limb may behave
  differently — has no independent cross-check at all.
- **Per-leg bias.** The gyro-minus-accelerometer offset differs by ~10 ms
  between legs, worth roughly 2.7 SI points. Small, but not zero, against a
  target of SI = 0.
- **Ankle angle** (foot + shank IMU, offline only). Waveform shape and landmark
  timing match published gait kinematics, but range of motion comes out
  ~17–20° against a literature 25–30°, most likely mounting compliance. Do not
  quote the magnitudes.
- **Toe-off** (`detect_toe_off_gyro`) is implemented, known to be wrong, and
  deliberately guarded. It reports ~35% stance per stride, which is not
  physiological. Nothing in the cost function uses it.

**If you are extending this**, the cheapest thing that would resolve most of
the above is a pair of FSR footswitches under the heels for one trial: a true
external reference on both legs simultaneously.

---

## Configuration

Copy `config/exo_symmetry_config.example.yml` to `exo_symmetry_config.yml` and edit:

```yaml
Subject:
  id: P001                                    # participant ID
  session: S001                               # session code
  base_dir: /path/to/HITLO_Data               # data root

Sensing:
  backend: trigno                             # trigno | polar
  detector: gyro                              # gyro | accel (defaults per backend)
  stream: TrignoIMU                           # LSL stream name (trigno only)

Cost:
  aim: Aim 1                                  # "Aim 1" (healthy) or "Aim 2" (stroke)
  sample_rate: 148                            # sensor rate (Hz): 148 Trigno, 200 Polar
  time: 90                                    # trial duration (seconds)
  signed: true                                # use signed SI (required for both aims)
  si_target: -10.0                            # fallback target (baseline phase overrides)
  trim_seconds: 3.0                           # trim from each end (steady-state window)

Optimization:
  index_csv: config/index_unified.csv         # x -> (R, theta, L0, attach) table
  n_parms: 1                                  # always 1 (the unified index)
  range: [[-1.0], [1.0]]                      # x spans DF-assist to PF-assist
  n_steps: 15                                 # total trials, including the ramp
  manual_ramp_trials: 5                       # first N trials follow ramp_sequence
  
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

- **v2.3.0** (current) — second sensor backend (Delsys Trigno Avanti) and
  gyroscope heel-strike detection, both selectable by config with the Polar +
  accelerometer path kept intact; unified stiffness index as the BO axis;
  console reworked around per-backend setup and preflight
- **v2.2.0** — (L0, attach) parameterization with R = 0.28 m fixed;
  bidirectional HILBO for stroke; baseline-relative targeting for healthy;
  hard torque constraints with safety fallback
- **v2.1.0** — switched to filter-then-diff at 50 Hz cutoff; tightened cluster-gap
- **v2.0.0** — refactored into library structure
- **v1.8.0** — cluster-keep-last added
- **v1.6.0** — jerk-based detection

---

## Hardware

- LegExoNET passive ankle exoskeleton (spring-pulley mechanism)
- Shank IMUs, one of:
  - **Delsys Trigno Avanti** — accel + gyro, ~148 Hz, published to LSL as one
    multiplexed stream by a bridge on the base-station machine. Channel labels
    carry side and segment (`left_shank_gyr_z`, `right_foot_acc_x`, …), which
    is how `hitlo.io` demultiplexes them. A third sensor on the foot is
    optional and is carried but not used by the cost function.
  - **Polar H10** ×2 — chest straps worn on the shanks with Coban wrap;
    accelerometer only, ~200 Hz, one LSL stream per side over BLE
- Mac laptop with LabRecorder, LSL, Python 3.9+

Sensor identifiers are configuration, not code — set them in
`exo_symmetry_config.yml` (Trigno: slot-to-label mapping in the bridge;
Polar: BLE device IDs). Always confirm the side mapping with
`./apps/verify_sides.py` or the console's shake test before a session.

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
