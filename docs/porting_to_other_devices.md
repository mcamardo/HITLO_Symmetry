# Porting HITLO_Symmetry to Other Devices

This guide covers what to change if you want to use this codebase for an exoskeleton or robotic device that's **not** the LegExoNET. It progresses from minimal (same mechanism, different limits) to substantial (different mechanism, different number of parameters).

If you only want to know *what to read*: the gait detection (`hitlo/detection.py`), symmetry computation (`hitlo/symmetry.py`), BO wrapper logic (`hitlo/hil_exo.py`), the Streamlit UI, and post-hoc analysis tools are all device-agnostic. They depend only on heel-strike timing from IMUs and on a 2-parameter (or N-parameter) optimization. The device-specific code is concentrated in **`hitlo/cost.py`** (the torque physics) and a few config values.

---

## Decision tree: what kind of port is this?

```
Is your device an ankle exoskeleton with the same spring-pulley mechanism?
│
├── YES → Port Level 1 (10 minutes): just change config values
│
└── NO ─→ Same number of parameters, different physics?
         │
         ├── YES → Port Level 2 (1-2 hours): swap the physics function
         │
         └── NO ─→ Different number/names of parameters?
                  │
                  └── YES → Port Level 3 (3-5 hours): generalize the
                            parameter handling
```

Pick the lowest level that applies to your device.

---

## Port Level 1: Same mechanism, different limits

**Use case:** A second LegExoNET unit, or a structurally identical exo with a different mechanical safety envelope (e.g. a participant-specific torque cap, or a different stretched-spring constant).

**What to change:** All the changes are in **one file**: `config/exo_symmetry_config.yml`.

```yaml
Subject:
  id: P049               # ← your subject
  session: S001
  base_dir: /Users/yourname/HITLO

Optimization:
  range:                 # ← parameter search box for your specific device
    - [0.24, 0.30]       # mins for [R, L0]
    - [0.35, 0.40]       # maxes
  pf_zone_deg: [2.0, 20.0]
  pf_torque_threshold: 4.0
  df_min_bo_nm: 0.0      # ← raise to require minimum DF assist during BO
  min_df_torque_nm: 20.0
```

**Hard torque cap** (currently `90 Nm`) lives in `hitlo/hil_exo.py` near the top:

```python
HARD_TORQUE_CAP = 90.0   # Nm — change for your device's mechanical limit
```

**Spring constant** (currently `12000 N/m`), **anchor angle** (`196°`), **foot segment length** (`0.335 m`), and other mechanism geometry live in `hitlo/cost.py:compute_exo_torque()`. If your device has different physical dimensions but the same mechanism, edit those constants there.

**That's it.** Don't change anything else. Run the same workflow as you would for any LegExoNET session.

---

## Port Level 2: Different physics, same parameter count (2 params)

**Use case:** A different ankle exoskeleton mechanism (cam-and-roller, leaf spring, pneumatic actuator, etc.) that still has 2 tunable parameters being optimized.

**What to change:** `hitlo/cost.py` — specifically `compute_exo_torque()`, and possibly `compute_torque_curve()` and `compute_spring_penalty()`.

### Step 1: Replace `compute_exo_torque()`

The current function takes `(ankle_angle_deg, R, L0)` and returns torque in Nm. Replace its body with your device's physics:

```python
def compute_exo_torque(ankle_angle_deg: float, param1: float, param2: float) -> float:
    """Torque produced by your device at a given joint angle.

    Args:
        ankle_angle_deg: joint angle in degrees (positive = PF, negative = DF
                         for ankle; adapt the convention for your joint)
        param1, param2: your two tunable device parameters
                        (rename in the function signature too if you want)

    Returns:
        Torque in Nm — positive = dorsiflexion assist, negative = resistance
    """
    # ── Your physics goes here ──
    # Example: simple linear spring with torque arm
    # k = 5000.0  # spring constant N/m
    # arm_length = 0.10  # m
    # delta = stretch_at_angle(ankle_angle_deg, param1, param2)
    # force = k * max(delta, 0)
    # torque = force * arm_length
    # return torque

    raise NotImplementedError("Implement your device's torque physics here")
```

### Step 2: Update the docstring at the top of the file

```python
"""
hitlo.cost — BO cost function for [your device name].
[describe what your two parameters represent]
"""
```

### Step 3: Verify with a test

Before you run a real experiment, sanity-check the new physics:

```python
# Quick sanity check (run in Python REPL)
from hitlo.cost import compute_exo_torque, compute_torque_curve

# At neutral angle (0°), what's the torque?
print(compute_exo_torque(0, param1=YOUR_VAL_1, param2=YOUR_VAL_2))

# Plot the torque curve over the joint range
import matplotlib.pyplot as plt
angles, torques = compute_torque_curve(param1=YOUR_VAL_1, param2=YOUR_VAL_2,
                                        angle_min=-30, angle_max=30)
plt.plot(angles, torques)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Joint angle (deg)'); plt.ylabel('Torque (Nm)')
plt.show()
```

The shape of this curve should match what your device actually does at known parameter settings. If a calibration test on the bench gives 25 Nm at -10° and your function predicts 5 Nm, your physics is wrong.

### Step 4: Adjust the angle range

The default joint angle range is `-30° to +30°`. If your joint operates differently (e.g., a knee that doesn't go beyond `0° to 110°`), update the defaults in:

```python
# hitlo/cost.py:89
def compute_torque_curve(param1: float, param2: float,
                         angle_min: float = -30.0,    # ← change
                         angle_max: float = 30.0,     # ← change
                         n_points: int = 100):
```

And in `hitlo/hil_exo.py:_is_safe_candidate()`:

```python
# hitlo/hil_exo.py:164
angles, torques = compute_torque_curve(
    param1, param2, angle_min=-30.0, angle_max=30.0, n_points=100)  # ← change
```

### Step 5: Update the safety zones

The "PF zone" and "DF check angle" are ankle-specific. For other joints, rename and reposition:

In `config/exo_symmetry_config.yml`:
```yaml
Optimization:
  pf_zone_deg: [2.0, 20.0]      # rename and rerange for your joint's "no-assist" zone
  df_check_angle_deg: -10.0     # rename for your joint's "assist target" angle
```

These are the **safety constraints** — the points the code checks to decide if a parameter combination is safe. Pick angles relevant to your joint's biomechanics.

---

## Port Level 3: Different number/names of parameters

**Use case:** Your device has 3, 4, or 5 tunable parameters (not just 2). Or your parameters have different names that you want to use throughout the code (e.g., `cam_radius`, `eccentricity`, `damping_coefficient`).

This is the most invasive port because the assumption "2 parameters named R and L0" is threaded through several files. Here's the systematic approach.

### Step 1: Pick your parameter names

Decide on N parameter names. Stick with them throughout. Example for a 3-parameter pneumatic device:

```
old:  R, L0
new:  pressure, valve_timing, plate_angle
```

### Step 2: Update the config

```yaml
Optimization:
  n_parms: 3              # ← number of dimensions
  param_names: ["pressure", "valve_timing", "plate_angle"]   # ← optional but useful
  range:
    - [0.0, 0.0, 5.0]     # mins for each param
    - [50.0, 100.0, 25.0] # maxes for each param
```

### Step 3: Update the physics function

```python
# hitlo/cost.py
def compute_exo_torque(ankle_angle_deg: float,
                        pressure: float,
                        valve_timing: float,
                        plate_angle: float) -> float:
    """Your N-parameter physics."""
    # ...
```

Or use `*params` if you want maximum flexibility:

```python
def compute_exo_torque(angle_deg: float, *params) -> float:
    pressure, valve_timing, plate_angle = params
    # ...
```

### Step 4: Update `compute_torque_curve()` and `compute_spring_penalty()`

Same idea — change the function signatures to accept your N parameters.

### Step 5: Find every usage of "R, L0" and update

Run this to find every place that needs updating:

```bash
cd HITLO_Symmetry
grep -rn "R, L0\|self._R\|self._L0" hitlo/ apps/ scripts/
```

You'll see ~20 hits across `cost.py`, `hil_exo.py`, `apps/run_experiment.py`, and `apps/diagnose_trial.py`. Walk through each one and update.

Key locations:

**`hitlo/cost.py`**:
- `SymmetryCost.set_params(R, L0)` → `set_params(*params)` or named kwargs
- `self._R, self._L0` → `self._params` (a tuple/array)
- All `compute_exo_torque(angle, R, L0)` calls → pass `*self._params`

**`hitlo/hil_exo.py`**:
- `_is_safe_candidate(R, L0, ...)` → `_is_safe_candidate(*params, ...)`
- All `R, L0 = candidate[0], candidate[1]` → `params = candidate`
- `_top_k_safe_fallback()` builds a 2D grid (`R_vals`, `L0_vals`); for higher dimensions, you'll need a sparser grid or a different sampling strategy. **Important**: a 50×50 grid is 2500 points and works in 2D. For 3D it'd be 125,000 (50³) — too many. For 3+D, replace the grid with random sampling weighted by EI, or use BoTorch's `optimize_acqf` with multiple restarts. See "Higher dimensions" below.

**`apps/run_experiment.py`**:
- Display logic — currently shows "R = 0.27, L₀ = 0.32". Update to show whatever your params are.

**`apps/diagnose_trial.py`**:
- The CLI prints "R, L0" in headers. Update for your params.

### Step 6: Higher-dimensional safe-fallback

The `_top_k_safe_fallback()` method in `hitlo/hil_exo.py` builds a `n_grid × n_grid` grid for 2D. For higher dimensions, replace this section:

```python
# Old (2D):
R_vals = np.linspace(range_[0, 0], range_[1, 0], n_grid)
L0_vals = np.linspace(range_[0, 1], range_[1, 1], n_grid)
RR, LL = np.meshgrid(R_vals, L0_vals)
grid_phys = np.column_stack([RR.ravel(), LL.ravel()])
```

with random-sample-then-rank:

```python
# New (any dimension):
n_samples = 5000  # Sobol or uniform random over the search box
sobol = scipy.stats.qmc.Sobol(d=n_parms, seed=42)
samples_norm = sobol.random(n_samples)
grid_phys = samples_norm * (range_[1] - range_[0]) + range_[0]
```

This gives you the same "evaluate acquisition, rank, walk down list" behavior in N dimensions.

### Step 7: Validate

Run `validate_refactor.py` (or write a similar one for your device) on a known dataset to confirm your refactored version produces the same costs as the original on test inputs.

---

## What stays the same regardless of your device

These are device-agnostic and require **no changes**:

| File | Why it doesn't need changes |
|---|---|
| `hitlo/detection.py` | IMU-based heel-strike detection — depends only on shank-mounted accel data, not on the exo |
| `hitlo/symmetry.py` | Step-time interleaving and SI computation — pure gait analysis |
| `hitlo/io.py` | XDF loading and BIDS filename helpers |
| Most of `hitlo/hil_exo.py` | The BO wrapper, LHS exploration, top-K fallback logic — all parameter-count-aware |
| `apps/run_experiment.py` UI structure | Streamlit layout, plot helpers, fragment patterns |
| `apps/diagnose_trial.py` | CLI flow |
| `scripts/analyze_experiment.py` | Post-hoc batch analysis |
| `apps/gp_viewer.py` | Interactive GP iteration browser |
| Detection thresholds in `hitlo/detection.py:DetectionConfig` | These tune to gait, not to the device — adjust per population (stroke vs. healthy vs. amputee) but not per exo |

If you find yourself editing any of these for a port, stop and reconsider — you're probably going about it the hard way.

---

## What to swap if your sensing approach is different

This guide assumes you're using **two shank-mounted IMU accelerometers** to detect heel strikes. If you're using something else:

| Sensing modality | What to swap |
|---|---|
| Foot pressure / GRF | Replace `hitlo/detection.py` with your foot-strike detector. Same downstream interface (return heel-strike timestamps). |
| Joint kinematics (motion capture) | Same — replace `detection.py` with a kinematic event detector. |
| EMG-based event detection | Same. |
| Different cost signal entirely (metabolic cost, joint moment, etc.) | Replace `hitlo/cost.py:SymmetryCost.extract_cost_from_file()` to compute your cost from the recording. The BO doesn't care what the cost represents. |

The architecture is intentionally modular here. Detection and cost computation are separate from the BO loop — you can swap either without touching the optimization or UI layers.

---

## Common pitfalls when porting

**1. Forgetting to validate the new physics.**
Your `compute_exo_torque()` is the heart of the safety system. If it's wrong, the BO might propose "safe" parameters that actually produce dangerous torques on the real device. Always test on the bench with calibration weights or a torque sensor before running on a participant.

**2. Mismatched angle conventions.**
LegExoNET uses positive = plantarflexion. Your device might use the opposite convention. The hard torque cap and PF/DF zones depend on this — getting it wrong means safety zones are inverted. Check by plotting the torque curve and confirming it goes the direction you expect.

**3. Hardcoded "ankle" terminology in error messages.**
The codebase prints things like "PF zone violation" and "DF assist torque". These are ankle-specific terms that won't make sense for a knee or hip device. Find-and-replace these for clarity (search: `PF\|DF\|ankle\|plantarflexion\|dorsiflexion`).

**4. Cost sign.**
The BO **minimizes** `|cost|`. If your cost signal is "metabolic cost" you want to minimize, fine. If it's "muscle activation reduction" you want to maximize, multiply by -1 in `extract_cost_from_file()`.

**5. Trial duration.**
60 seconds works for steady-state gait. If your task is shorter (e.g., a 3-second sit-to-stand), update `Cost.time` in the config and consider whether your detection algorithm has enough data points to work in that window.

---

## When to ask vs. when to fork

**Ask Mac first if:**
- You're not sure which port level applies to your device
- The hard torque cap concept doesn't translate (e.g., your device is force-controlled, not torque-controlled)
- You want to add a new cost signal type but aren't sure how to integrate it
- You're porting to a non-walking task (cycling, sit-to-stand, reaching)

**Just fork and edit if:**
- You're doing a Level 1 or Level 2 port for a similar ankle device
- The mapping is straightforward and you understand the code
- You're prototyping and willing to throw away a branch if needed

If you fork, please open an issue on the original repo to let me know — even if you don't contribute back, knowing what people are trying to use this for helps prioritize what to abstract next.

---

## What I'd want this codebase to become eventually

If multiple people end up porting this to different devices, the natural next step is to extract the device-specific bits into a `Device` class hierarchy:

```python
class Device(ABC):
    @abstractmethod
    def compute_torque(self, angle, *params): ...
    @abstractmethod
    def get_safety_constraints(self): ...

class LegExoNET(Device):
    def compute_torque(self, angle, R, L0): ...

class YourDevice(Device):
    def compute_torque(self, angle, p1, p2, p3): ...
```

Then everything else in `hitlo/` would take a `Device` object and call its methods. That's the clean way. **But** I'm not doing that refactor pre-emptively — premature abstraction is worse than no abstraction. If you're the second person to port this, please get in touch and we'll do it together, informed by your actual use case.

---

*Questions? File an issue or email Mac Camardo.*
