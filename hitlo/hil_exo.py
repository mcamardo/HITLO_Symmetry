"""
hitlo.hil_exo — experiment driver for the unified-index parameterization.

Version 3.0.0 — ONE parameter.

BO optimizes a single number, x in [-1, +1]:
    x = -1   max dorsiflexor stiffness (limited by the 24-band ceiling)
    x =  0   zero torque
    x = +1   max plantarflexor stiffness (limited by the 80 Nm cap)

Each x maps to a row of index_unified.csv holding the four physical
parameters the operator sets on the device: (R, theta, L0, attachment_ratio).

WHY THERE IS NO SAFETY CHECKING IN THIS FILE ANYMORE
Every row of the index table already passed sign purity, the torque cap, the
engagement window, and the edge-cliff filter — in build_index_unified.m,
against the validated forward model. BO can only ever propose a row that
exists, so every configuration it can reach is safe by construction. The old
_is_safe_candidate / _top_k_safe_fallback / _get_safe_bo_suggestion ladder
existed because the old (L0, attach) space was continuous and unfiltered.
It is gone. If you change a cap or a filter, regenerate the CSV — do not add
runtime checks here, or the two will drift apart.

HOW THE NEXT TRIAL IS CHOSEN
The acquisition function is evaluated ONLY at the x values that exist in the
table, and the argmax is taken. Nothing is interpolated and nothing needs
snapping after the fact: BO cannot name a configuration that was never built.
"""

from typing import Dict
import numpy as np

try:
    from HIL.optimization.BO import BayesianOptimization
except ImportError as e:
    raise ImportError(
        "HITLO_Symmetry requires HIL_toolkit. Install it with:\n"
        "    git clone https://github.com/UICRRL/HIL_toolkit.git\n"
        "    cd HIL_toolkit && pip install -e .\n"
        f"Original error: {e}"
    )

from hitlo.index_unified import IndexTable


class HIL_Exo:
    """Runs the HITLO experiment over the 1-D unified index.

    Args:
        args: Experiment config dict (from exo_symmetry_config.yml). Expects
              'Optimization' and 'Cost'. Optimization must carry
              'index_csv' (path to index_unified.csv), 'n_steps',
              'manual_ramp_trials', 'ramp_sequence' (list of x values),
              and 'model_save_path'.
        cost_extractor: object with .cost_name and
                        .extract_cost_from_file(trial_num).
    """

    def __init__(self, args: Dict, cost_extractor) -> None:
        self.n = 0
        self.args = args
        self.cost = cost_extractor
        opt = args["Optimization"]

        self.table = IndexTable(opt["index_csv"])

        self.NORMALIZATION = opt.get("normalize", True)
        self.x = np.array([])          # planned x per trial
        self.x_opt = np.array([])      # x values actually run
        self.y_opt = np.array([])      # costs
        self.signed = args.get("Cost", {}).get("signed", False)
        self.si_target = float(args.get("Cost", {}).get("si_target", 0.0))

        # ONE source of truth for where the ramp ends and BO begins. The old
        # code had n_exploration driving the BO trigger while
        # manual_ramp_trials drove generation — two keys, one boundary,
        # guaranteed to disagree eventually.
        self.n_ramp = int(opt.get("manual_ramp_trials", 0))

        self._start_optimization(opt)

        self._bo_direction_str = (
            f"|SI - {self.si_target:+.1f}|" if self.signed else "cost"
        )

        print("HIL_Exo initialized — unified index (1 parameter)")
        print(f"   Index table: {self.table.path} ({len(self.table)} levels)")
        print(f"   Search space: x in [-1, +1], "
              f"{len(self.table.x_values)} reachable values")
        print(f"   BO direction: MINIMIZING {self._bo_direction_str} "
              f"(signed={self.signed}, si_target={self.si_target:+.1f})")
        print("   Safety: enforced at table build time, not at runtime")

    # =======================================================================
    # Normalization (x in [-1,1] <-> [0,1] for GP numerical stability)
    # =======================================================================

    def _normalize_x(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float).reshape(-1, 1) + 1.0) / 2.0

    def _denormalize_x(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=float).reshape(-1, 1) * 2.0 - 1.0,
                       -1.0, 1.0)

    def _mean_normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize y for GP input.

        Signed mode: |y - si_target|, so BO minimizes distance from the target
        asymmetry. si_target=0 drives SI toward symmetry (Aim 2, stroke);
        si_target=-10 drives toward a target asymmetry (Aim 1, neurotypical).

        Mean-center and std-scale for GP stability, then negate so BoTorch
        maximization equals cost minimization.
        """
        y = np.asarray(y, dtype=float)
        if self.signed:
            y = np.abs(y - self.si_target)
        y = (y - np.mean(y)) / (np.std(y) + 1e-8)
        return -y

    # =======================================================================
    # BO setup
    # =======================================================================

    def _start_optimization(self, opt: Dict) -> None:
        if self.NORMALIZATION:
            rng = np.array([[0.0], [1.0]])
        else:
            rng = np.array([[-1.0], [1.0]])
        self.BO = BayesianOptimization(
            n_parms=1,
            range=rng,
            model_save_path=opt["model_save_path"],
        )

    def _generate_initial_parameters(self) -> None:
        """Load the manual ramp. BO generates everything after it, on demand."""
        opt = self.args["Optimization"]

        if self.n_ramp > 0 and "ramp_sequence" in opt:
            ramp = np.asarray(opt["ramp_sequence"], dtype=float).ravel()
            # Snap the ramp too — a hand-written config value like -0.5 is not
            # necessarily a row, and the operator must set a real configuration.
            snapped = np.array([self.table.snap(v) for v in ramp])
            for raw, snap in zip(ramp, snapped):
                if not np.isclose(raw, snap):
                    print(f"   ramp x={raw:+.4f} snapped to {snap:+.4f}")
            self.x = snapped.reshape(-1, 1)
            print(f"Loaded {len(self.x)} ramp trials")
            print(f"   Trials 1-{self.n_ramp}: manual ramp")
            print(f"   Trials {self.n_ramp + 1}-{opt['n_steps']}: "
                  f"Bayesian Optimization (on demand)")
        else:
            self.x = np.empty((0, 1))
            print(f"No manual ramp. BO generates all {opt['n_steps']} trials.")

    # =======================================================================
    # Candidate selection: acquisition evaluated on the real rows only
    # =======================================================================

    def _next_x_from_table(self, fallback: float) -> float:
        """Pick the next x by maximizing acquisition over the table's rows.

        Evaluating only at reachable x means the result needs no snapping and
        can never be a configuration that was not built and validated.

        Falls back to snapping BO's own continuous suggestion if the
        acquisition cannot be built (e.g. too few points to fit the GP).
        """
        try:
            import torch
            from botorch.acquisition import qNoisyExpectedImprovement
            from botorch.sampling import IIDNormalSampler

            cand = self.table.x_values.reshape(-1, 1)
            cand_n = self._normalize_x(cand) if self.NORMALIZATION else cand
            train = (self._normalize_x(self.x_opt) if self.NORMALIZATION
                     else self.x_opt.reshape(-1, 1))

            sampler = IIDNormalSampler(sample_shape=torch.Size([200]), seed=1234)
            acq = qNoisyExpectedImprovement(
                self.BO.model,
                torch.tensor(train, dtype=torch.float64),
                sampler=sampler,
            )

            vals = np.zeros(len(cand_n))
            with torch.no_grad():
                for s in range(0, len(cand_n), 100):
                    e = min(s + 100, len(cand_n))
                    batch = torch.tensor(cand_n[s:e],
                                         dtype=torch.float64).unsqueeze(1)
                    vals[s:e] = acq(batch).numpy()

            # Discourage re-running a level already tested. Not forbidden —
            # a repeat is legitimate if the GP really wants it — but with only
            # ~10 BO trials, spending one on a known point is usually waste.
            for i, xv in enumerate(self.table.x_values):
                if np.any(np.isclose(self.x_opt.ravel(), xv)):
                    vals[i] = -np.inf

            if np.all(np.isneginf(vals)):
                print("   every level already tested — allowing a repeat")
                return float(self.table.snap(fallback))

            best = int(np.argmax(vals))
            chosen = float(self.table.x_values[best])
            print(f"   acquisition argmax over {len(cand)} levels -> "
                  f"x={chosen:+.4f} (EI={vals[best]:.4f})")
            return chosen

        except Exception as e:
            snapped = float(self.table.snap(fallback))
            print(f"   acquisition scan failed ({type(e).__name__}: {e})")
            print(f"   falling back to snapping BO's suggestion: "
                  f"{fallback:+.4f} -> {snapped:+.4f}")
            return snapped

    # =======================================================================
    # Reporting
    # =======================================================================

    def _best_so_far_idx(self) -> int:
        if self.signed:
            return int(np.argmin(np.abs(self.y_opt - self.si_target)))
        return int(np.argmin(self.y_opt))

    def print_trial_parameters(self, trial_num: int, x_val: float) -> None:
        """Print the four physical parameters. x is meaningless at the bench."""
        r = self.table.row(float(x_val))
        n_steps = self.args["Optimization"]["n_steps"]

        print("\n" + "=" * 62)
        print(f"   TRIAL {trial_num}/{n_steps}", end="")
        print("   [manual ramp]" if trial_num <= self.n_ramp
              else "   [Bayesian optimization]")
        print("=" * 62)
        print(f"   index x = {r['x']:+.4f}"
              f"   ({'PF' if r['direction'] > 0 else 'DF' if r['direction'] < 0 else 'ZERO'})")
        print(f"   stiffness {r['stiff_Nm_per_rad']:+.1f} Nm/rad"
              f"   |  dose {r['dose_Nm']:+.2f} Nm in ROM"
              f"   |  peak {r['peak_full_Nm']:.2f} Nm full range")
        print("-" * 62)

        if r["is_zero"]:
            print("   ZERO TORQUE CONDITION")
            print("   This is NOT device-off. The 24 bands are always on, so")
            print("   the controller must actively cancel them:")
            print("       tau_cable = 0 - tau_band(theta)")
        else:
            print(f"{'SET ON DEVICE:':^62}")
            print(f"   R      = {r['R']:.4f} m   ({r['R'] * 100:.2f} cm)  anchor distance")
            print(f"   theta  = {r['theta']:.2f} deg              anchor angle")
            print(f"   L0     = {r['L0']:.4f} m   ({r['L0'] * 100:.2f} cm)  slack length")
            print(f"   attach = {r['attach']:+.4f}                foot attachment ratio")
            print(f"   (engages at {r['engage_deg']:+.1f} deg)")
            if r["direction"] < 0:
                print("   DF: controller renders tau_cable = tau_desired - tau_band(theta)")
        print("-" * 62 + "\n")

    def _log_trial(self, prefix: str, x_val: float, cost_value: float) -> None:
        r = self.table.row(float(x_val))
        detail = (f"x={r['x']:+.4f} (dose {r['dose_Nm']:+.2f} Nm, "
                  f"{r['stiff_Nm_per_rad']:+.1f} Nm/rad)")
        if self.signed:
            dist = abs(cost_value - self.si_target)
            print(f"\n{prefix}: SI = {cost_value:+.4f}  |  "
                  f"|SI - target| = {dist:.4f}  |  {detail}")
        else:
            print(f"\n{prefix}: Cost = {cost_value:.4f}  |  {detail}")

    # =======================================================================
    # Main loop
    # =======================================================================

    def start(self):
        opt = self.args["Optimization"]

        if self.n == 0:
            print(f"\n{'=' * 70}")
            print("STARTING HIL EXOSKELETON OPTIMIZATION (unified index)")
            print(f"Cost function:  {self.cost.cost_name}")
            print(f"Search space:   1 parameter, x in [-1, +1]")
            print(f"Index levels:   {len(self.table)} (from {self.table.path.name})")
            print(f"Total trials:   {opt['n_steps']}  "
                  f"({self.n_ramp} ramp + {opt['n_steps'] - self.n_ramp} BO)")
            print(f"BO direction:   MINIMIZING {self._bo_direction_str}")
            print(f"{'=' * 70}\n")
            self._generate_initial_parameters()

        while self.n < opt["n_steps"]:
            trial_num = self.n + 1
            x_val = float(self.x[self.n, 0])
            self.print_trial_parameters(trial_num, x_val)

            print("INSTRUCTIONS:")
            print("  1. Set the four parameters above on the device")
            print("  2. Start LabRecorder:")
            print("     - Update streams, check 'polar accel left' AND 'polar accel right'")
            print(f"     - Set filename: trial_{trial_num:02d}.xdf")
            print("     - Click 'Start'")
            print(f"  3. Subject walks for {self.args['Cost']['time']}s")
            print("  4. LabRecorder: Click 'Stop'")
            print(f"\n  Press ENTER when trial_{trial_num:02d}.xdf is saved...")
            input()

            cost_value = self.cost.extract_cost_from_file(trial_num)

            if cost_value is None or np.isnan(cost_value):
                print("Cost extraction failed.")
                if input("Retry this trial? (Y/n): ").lower() != "n":
                    continue
                self.n += 1
                continue

            # Record the x that was ACTUALLY set, which is always a table row.
            if len(self.x_opt) < 1:
                self.x_opt = np.array([[x_val]])
                self.y_opt = np.array([cost_value])
            else:
                self.x_opt = np.concatenate((self.x_opt, [[x_val]]))
                self.y_opt = np.concatenate((self.y_opt, [cost_value]))

            self._log_trial("Recorded", x_val, cost_value)
            self.n += 1

            # Fit the GP and choose the next level.
            if self.n_ramp <= self.n < opt["n_steps"]:
                print(f"\nRunning Bayesian Optimization "
                      f"(minimizing {self._bo_direction_str})...")

                if self.NORMALIZATION:
                    raw = self.BO.run(
                        self._normalize_x(self.x_opt).reshape(self.n, -1),
                        self._mean_normalize_y(self.y_opt).reshape(self.n, -1),
                    )
                    raw = float(self._denormalize_x(raw).ravel()[0])
                else:
                    y_for_bo = (-np.abs(self.y_opt - self.si_target)
                                if self.signed else -self.y_opt)
                    raw = self.BO.run(
                        self.x_opt.reshape(self.n, -1),
                        y_for_bo.reshape(self.n, -1),
                    )
                    raw = float(np.asarray(raw).ravel()[0])

                next_x = self._next_x_from_table(raw)
                self.x = np.concatenate((self.x, [[next_x]]), axis=0)

            best_idx = self._best_so_far_idx()
            self._log_trial("Best so far",
                            float(self.x_opt[best_idx, 0]),
                            float(self.y_opt[best_idx]))

            if self.n < opt["n_steps"]:
                input("\nPress ENTER to continue to next trial...")

        print("\n" + "=" * 70)
        print("   OPTIMIZATION COMPLETE")
        print("=" * 70)
        best_idx = self._best_so_far_idx()
        r = self.table.row(float(self.x_opt[best_idx, 0]))

        if self.signed:
            print(f"  SI:            {self.y_opt[best_idx]:+.4f}")
            print(f"  |SI - target|: {abs(self.y_opt[best_idx] - self.si_target):.4f}"
                  f"  (target {self.si_target:+.1f})")
        else:
            print(f"  Cost:          {self.y_opt[best_idx]:.4f}")

        print(f"\n  Best index:    x = {r['x']:+.4f}"
              f"  ({'PF' if r['direction'] > 0 else 'DF' if r['direction'] < 0 else 'ZERO'})")
        print(f"  Stiffness:     {r['stiff_Nm_per_rad']:+.1f} Nm/rad")
        print(f"  Dose in ROM:   {r['dose_Nm']:+.2f} Nm")
        if not r["is_zero"]:
            print(f"  R = {r['R']:.4f} m,  theta = {r['theta']:.2f} deg,  "
                  f"L0 = {r['L0']:.4f} m,  attach = {r['attach']:+.4f}")

        # Which arm won. Worth logging per subject: if DF keeps winning, the
        # single GP is under-resolving the arm that matters and the two-GP
        # split becomes worth building.
        n_pf = int(np.sum(self.x_opt.ravel() > 0))
        n_df = int(np.sum(self.x_opt.ravel() < 0))
        n_zero = int(np.sum(np.isclose(self.x_opt.ravel(), 0.0)))
        print(f"\n  Trials by arm: {n_df} DF, {n_zero} zero, {n_pf} PF")

        print(f"\n  Model saved to: {self.args['Optimization']['model_save_path']}")


__all__ = ["HIL_Exo"]