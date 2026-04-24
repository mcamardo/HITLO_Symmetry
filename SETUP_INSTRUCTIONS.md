# Setup instructions — starting from a fresh download

You just downloaded `HITLO_Symmetry` to your Mac. Here's how to get it running.

## 1. Move to your working directory

```bash
mv ~/Downloads/HITLO_Symmetry ~/Projects/HITLO_Symmetry
cd ~/Projects/HITLO_Symmetry
```

(Or wherever you want to keep it.)

## 2. Verify the file structure

```bash
find . -type f | sort
```

You should see:

```
./.gitignore
./LICENSE
./README.md
./SETUP_INSTRUCTIONS.md
./docs/workflow.md
./hitlo/__init__.py
./hitlo/hil_exo.py
./hitlo/cost.py
./hitlo/detection.py
./hitlo/io.py
./hitlo/symmetry.py
./requirements.txt
```

## 3. Install HIL_toolkit (external dependency)

Put it in a SIBLING folder, NOT inside HITLO_Symmetry:

```bash
cd ~/Projects
git clone https://github.com/UICRRL/HIL_toolkit.git
cd HIL_toolkit
pip install -e .
```

## 4. Install HITLO_Symmetry requirements

```bash
cd ~/Projects/HITLO_Symmetry
pip install -r requirements.txt
```

## 5. Verify the library imports work

```bash
python3 -c "
from hitlo import DetectionConfig, detect_heelstrikes_full
from hitlo.cost import SymmetryCost
from hitlo.hil_exo import HIL_Exo
print('✅ All imports work')
print('Default detection config:', DetectionConfig())
"
```

If you see `✅ All imports work`, you're ready.

If you get `ImportError: HITLO requires HIL_toolkit`, go back to step 3.

## 6. Validate against your existing pipeline (recommended)

Before throwing out your old code, run this sanity check on a known trial.
It confirms the new library gives the same cost as your old `symmetry_cost.py`:

```python
# validate_refactor.py — drop this in your HITLO folder and run
import numpy as np

# Old pipeline
from symmetry_cost import SymmetryCost as OldCost
old = OldCost(trial_data_dir="/path/to/your/xdfs", signed=True, trim_seconds=3.0)
old_cost = old.extract_cost_from_file(
    trial_num=7,
    filename="sub-P048_ses-S001_task-Default_run-007_eeg.xdf")

# New pipeline
import sys
sys.path.insert(0, "/Users/maccamardo/Projects/HITLO_Symmetry")
from hitlo.cost import SymmetryCost as NewCost
new = NewCost(
    trial_data_dir="/path/to/your/xdfs",
    subject_id="P048", session="S001",
    signed=True, trim_seconds=3.0)
new_cost = new.extract_cost_from_file(trial_num=7)

print(f"OLD: {old_cost:.4f}")
print(f"NEW: {new_cost:.4f}")
print(f"MATCH: {abs(old_cost - new_cost) < 0.01}")
```

If the two costs match → the refactor is working. Proceed with Phase 2.
If they differ by more than rounding → stop and share the numbers with Claude,
we'll debug.

## 7. Initialize the git repo (for GitHub)

```bash
cd ~/Projects/HITLO_Symmetry
git init
git branch -m main
git add .
git commit -m "Initial commit: refactored library structure (v2.0.0)

- hitlo/ library with single-source-of-truth detection pipeline
- Consolidated from symmetry_cost.py v1.8.0, symmetry_diagnose.py v1.7.3, ui.py v1.11.0
- Added HIL_toolkit wrapper (hitlo/hil_exo.py) to isolate dependency
- README with installation, workflow, citations
- MIT license"
```

## 8. Create private GitHub repo

On github.com:
1. Click **+** → **New repository**
2. Name: `HITLO_Symmetry`
3. Description: "Symmetry-based cost function for HITLO exoskeleton optimization"
4. Visibility: **Private**
5. Do NOT add README/gitignore/license (you already have them)
6. **Create repository**

Then push:

```bash
git remote add origin git@github.com:<your-username>/HITLO_Symmetry.git
git push -u origin main
```

## 9. Add collaborators

On GitHub → Settings → Collaborators → add by username:
- Dr. Patton
- Dr. Kim
- Any lab members who need access

---

## What you DON'T have yet (on purpose)

The `apps/`, `scripts/`, `config/`, and most of `docs/` folders are empty.
That's Phase 2 — we'll build those next once you confirm the library works
against your real data.

The files you DO have are enough to:
- Run the detection pipeline in a script or notebook
- Compute symmetry from XDF files programmatically
- Use the BO engine via `hitlo.hil_exo.HIL_Exo`

But not enough to:
- Run the Streamlit UI (needs apps/run_experiment.py)
- Run the diagnostic plotter (needs apps/diagnose_trial.py)
- Connect Polar sensors (needs apps/collect_sensors.py)

---

## Need help?

Come back to the Claude conversation and say "library is set up, ready for Phase 2."
