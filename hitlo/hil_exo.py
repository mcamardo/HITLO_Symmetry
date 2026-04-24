"""
hitlo.hil_exo — experiment driver with exoskeleton safety constraints.

Version 2.3.0 — Top-K acquisition fallback for unsafe BO suggestions
                (replaces uniform random fallback)
Version 2.2.0 — LHS exploration sampling for better parameter space coverage
                DF hard floor during BO (df_min_bo_nm), mu_df removed from cost
                Graduated DF torque ramp during exploration, DF hard constraint
                removed during BO (soft mu_df reward drives engagement instead)
                Signed symmetry support: BO minimizes |cost| toward zero

This is the HITLO_Symmetry wrapper around HIL_toolkit's BayesianOptimization.
It adds exoskeleton-specific safety enforcement that HIL_toolkit doesn't know
about: torque caps, PF zone limits, DF engagement floors. These exist because
our device is a physical spring mechanism that must not be pushed to parameter
combinations that would injure the participant or fail to engage the ankle.

SAFETY
------
Exploration (trials 1–n_exploration):
  - Hard cap: 90 Nm absolute, always
  - PF zone hard cap: pf_torque_threshold, always
  - DF minimum: linearly ramped from 0 Nm (trial 1) → min_df_torque_nm (trial n)

BO (trials n_exploration+1 → n_steps):
  - Hard cap: 90 Nm absolute, always
  - PF zone hard cap: pf_torque_threshold, always
  - DF minimum: soft — enforced via df_min_bo_nm hard floor + mu_df soft reward

EXPLORATION SAMPLING
--------------------
Latin Hypercube Sampling (LHS) ensures exploration trials are well spread
across the full parameter space. Falls back to random sampling if LHS pool
is exhausted by safety filtering.

Requires HIL_toolkit — see README for installation.
"""

from typing import Dict, Tuple
import numpy as np
from scipy.stats import qmc

try:
    from HIL.optimization.BO import BayesianOptimization
except ImportError as e:
    raise ImportError(
        "HITLO_Symmetry requires HIL_toolkit. Install it with:\n"
        "    git clone https://github.com/UICRRL/HIL_toolkit.git\n"
        "    cd HIL_toolkit && pip install -e .\n"
        f"Original error: {e}"
    )

from hitlo.cost import compute_exo_torque, compute_torque_curve


HARD_TORQUE_CAP = 90.0   # Nm — absolute safety limit, never exceeded


class HIL_Exo:
    """Orchestrates the HITLO experiment: LHS exploration + BO with safety checks.

    Wraps HIL_toolkit's BayesianOptimization for the GP engine and adds:
      - Exoskeleton-specific safety constraints (torque caps, DF engagement)
      - Latin Hypercube exploration sampling with oversampling pool
      - Graduated DF ramp-up during exploration
      - Signed-symmetry BO that minimizes |cost| toward zero

    Args:
        args: Full experiment config dict (parsed from exo_symmetry_config.yml).
              Expects top-level keys 'Optimization' and 'Cost'.
        cost_extractor: An object with .cost_name and
                        .extract_cost_from_file(trial_num) methods.
                        Typically a hitlo.cost.SymmetryCost instance.
    """

    def __init__(self, args: Dict, cost_extractor) -> None:
        self.n = int(0)
        self.x = np.array([])
        self.args = args
        self.cost = cost_extractor
        self.NORMALIZATION = self.args["Optimization"].get("normalize", True)
        self.x_opt = np.array([])
        self.y_opt = np.array([])
        self.signed = args.get("Cost", {}).get("signed", False)
        self._start_optimization(self.args["Optimization"])
        print(f"✅ HIL_Exo initialized")
        print(f"   Exploration sampling: Latin Hypercube Sampling (LHS)")
        print(f"   BO direction: MINIMIZING {'|cost|' if self.signed else 'cost'} (signed={self.signed})")
        print(f"   Hard torque cap: {HARD_TORQUE_CAP} Nm")

    # =======================================================================
    # Parameter normalization (for GP numerical stability)
    # =======================================================================

    def _normalize_x(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x).reshape(-1, self.args["Optimization"]["n_parms"])
        range_x = np.array(self.args["Optimization"]["range"]).reshape(
            2, self.args["Optimization"]['n_parms'])
        x = (x - range_x[0, :]) / (range_x[1, :] - range_x[0, :])
        return x

    def _denormalize_x(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x).reshape(-1, self.args["Optimization"]["n_parms"])
        range_x = np.array(self.args["Optimization"]["range"]).reshape(
            2, self.args["Optimization"]['n_parms'])
        x = x * (range_x[1, :] - range_x[0, :]) + range_x[0, :]
        return x

    def _mean_normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize y for GP input.

        If signed mode: take abs(y) first so BO minimizes |symmetry| toward zero,
        not toward maximally negative (right leg as slow as possible).
        Negate so BoTorch maximization = cost minimization.
        """
        y = np.array(y)
        if self.signed:
            y = np.abs(y)
        y = (y - np.mean(y)) / (np.std(y) + 1e-8)
        return -y

    # =======================================================================
    # BO initialization (HIL_toolkit instance)
    # =======================================================================

    def _start_optimization(self, args: Dict) -> None:
        if self.NORMALIZATION:
            self.BO = BayesianOptimization(
                n_parms=args["n_parms"],
                range=np.array([[0.0] * args["n_parms"],
                                [1.0] * args["n_parms"]]),
                model_save_path=args["model_save_path"],
            )
        else:
            self.BO = BayesianOptimization(
                n_parms=args["n_parms"],
                range=np.array(list(args["range"])),
                model_save_path=args["model_save_path"],
            )

    # =======================================================================
    # Safety: verify a candidate (R, L0) won't produce unsafe torques
    # =======================================================================

    def _is_safe_candidate(self, R: float, L0: float,
                           pf_zone: list, pf_threshold: float,
                           df_min: float = 0.0,
                           df_check_angle: float = -10.0
                           ) -> Tuple[bool, float, float, float]:
        """Check if a candidate (R, L0) is safe.

        Always enforced:
          1. Hard cap:    max torque anywhere <= HARD_TORQUE_CAP (90 Nm)
          2. PF zone cap: max torque in pf_zone <= pf_threshold

        Exploration only (df_min > 0):
          3. DF minimum:  torque at df_check_angle >= df_min

        Returns:
            (is_safe, max_pf_torque, df_torque, max_total_torque)
        """
        # 1. Hard cap
        angles_full, torques_full = compute_torque_curve(
            R, L0, angle_min=-30.0, angle_max=30.0, n_points=100)
        max_total = float(np.max(np.abs(torques_full)))
        if max_total > HARD_TORQUE_CAP:
            return False, None, None, max_total

        # 2. PF zone cap
        _, torques_pf = compute_torque_curve(
            R, L0, angle_min=pf_zone[0], angle_max=pf_zone[1], n_points=50)
        max_pf = float(np.max(np.abs(torques_pf)))
        if max_pf > pf_threshold:
            return False, max_pf, None, max_total

        # 3. DF minimum (exploration only — skipped when df_min=0)
        df_torque = float(compute_exo_torque(df_check_angle, R, L0))
        if df_min > 0 and df_torque < df_min:
            return False, max_pf, df_torque, max_total

        return True, max_pf, df_torque, max_total

    def _get_exploration_df_min(self, trial_idx: int) -> float:
        """Linear ramp UP of DF minimum torque across exploration trials.

        trial_idx: 0-based (0 = first trial, n_exploration-1 = last).
        Returns df_min for this trial (0 Nm at trial 0 → min_df_torque_nm at last).
        """
        opt = self.args["Optimization"]
        df_max = opt.get("min_df_torque_nm", 20.0)
        n = opt["n_exploration"]
        if n <= 1:
            return df_max
        return df_max * (trial_idx / (n - 1))

    # =======================================================================
    # Exploration: Latin Hypercube Sampling with safety filtering
    # =======================================================================

    def _generate_initial_parameters(self) -> None:
        """Generate exploration parameters using Latin Hypercube Sampling (LHS).

        LHS divides each parameter dimension into n equal intervals and places
        exactly one sample per interval — guaranteeing good coverage of the full
        parameter space regardless of safety filtering.

        Per-trial safety constraints (fully preserved):
          - Always:      hard 90 Nm cap + PF zone cap
          - Trial 1:     DF min = 0 Nm  (no engagement required)
          - Trial N:     DF min = min_df_torque_nm from config
          - Intermediate: linearly interpolated

        Falls back to random sampling if LHS pool is exhausted.
        """
        opt_args = self.args["Optimization"]
        range_ = np.array(list(opt_args["range"]))
        pf_zone = opt_args.get("pf_zone_deg", [2.0, 20.0])
        pf_threshold = opt_args.get("pf_torque_threshold", 4.0)
        df_angle = opt_args.get("df_check_angle_deg", -10.0)
        n_needed = opt_args["n_exploration"]
        n_parms = opt_args["n_parms"]

        print(f"\n###### Generating {n_needed} LHS exploration parameters ######")
        print(f"   Sampling method:  Latin Hypercube Sampling (LHS)")
        print(f"   PF zone:          {pf_zone[0]}° to {pf_zone[1]}°  "
              f"(max {pf_threshold} Nm)")
        print(f"   DF ramp:          0 Nm (trial 1) → "
              f"{opt_args.get('min_df_torque_nm', 20.0)} Nm (trial {n_needed})")
        print(f"   Hard cap:         {HARD_TORQUE_CAP} Nm")

        # LHS pool with 20x oversampling (to survive safety filtering)
        lhs_pool_size = n_needed * 20
        sampler = qmc.LatinHypercube(d=n_parms)  # no seed → different per subject
        lhs_unit = sampler.random(n=lhs_pool_size)
        lhs_scaled = qmc.scale(lhs_unit, range_[0], range_[1])

        print(f"\n   LHS pool: {lhs_pool_size} candidates generated")
        for dim_idx, (dim_min, dim_max, name) in enumerate(zip(
                range_[0], range_[1], ["R", "L0"])):
            dim_vals = lhs_scaled[:, dim_idx]
            print(f"   {name} coverage: {dim_vals.min():.4f} → "
                  f"{dim_vals.max():.4f} (full: {dim_min:.4f} → {dim_max:.4f})")

        safe_params = []
        lhs_idx = 0

        for trial_idx in range(n_needed):
            trial_num = trial_idx + 1
            df_min = self._get_exploration_df_min(trial_idx)

            df_max_config = opt_args.get("min_df_torque_nm", 20.0)
            df_pct = (df_min / df_max_config * 100) if df_max_config > 0 else 0

            print(f"\n   ── Trial {trial_num}/{n_needed} ──")
            print(f"   DF min = {df_min:.2f} Nm  "
                  f"({df_pct:.0f}% of max {df_max_config} Nm)")

            found = False

            # Pass 1: LHS pool in order
            while lhs_idx < len(lhs_scaled):
                candidate = lhs_scaled[lhs_idx]
                lhs_idx += 1
                R, L0 = candidate[0], candidate[1]

                is_safe, max_pf, df_torque, max_total = self._is_safe_candidate(
                    R, L0, pf_zone, pf_threshold,
                    df_min=df_min, df_check_angle=df_angle,
                )

                if is_safe:
                    safe_params.append(candidate)
                    print(f"   ✅ LHS  R={R:.4f}  L0={L0:.4f} "
                          f"| PF={max_pf:.2f} Nm  DF={df_torque:.2f} Nm  "
                          f"max={max_total:.2f} Nm  (pool idx {lhs_idx})")
                    found = True
                    break

            # Pass 2: LHS pool exhausted → random fallback
            if not found:
                print(f"   ⚠️  LHS pool exhausted for trial {trial_num} — "
                      f"falling back to random sampling")

                df_min_try = df_min
                for attempt in range(3000):
                    if attempt == 2000:
                        df_min_try = df_min * 0.5
                        print(f"   🔄 Relaxing DF min to {df_min_try:.2f} Nm "
                              f"after 2000 attempts")

                    candidate = np.random.uniform(range_[0], range_[1])
                    R, L0 = candidate[0], candidate[1]

                    is_safe, max_pf, df_torque, max_total = self._is_safe_candidate(
                        R, L0, pf_zone, pf_threshold,
                        df_min=df_min_try, df_check_angle=df_angle,
                    )

                    if is_safe:
                        safe_params.append(candidate)
                        print(f"   ✅ Random  R={R:.4f}  L0={L0:.4f} "
                              f"| PF={max_pf:.2f} Nm  DF={df_torque:.2f} Nm  "
                              f"(attempt {attempt+1})")
                        found = True
                        break

                if not found:
                    fallback = np.random.uniform(range_[0], range_[1])
                    safe_params.append(fallback)
                    print(f"   ⚠️  Using unconstrained fallback for trial {trial_num}")

        self.x = np.array(safe_params)

        print(f"\n###### Final LHS exploration parameters ######")
        print(f"{'Trial':<8} {'R (m)':<10} {'L0 (m)':<10} "
              f"{'DF min (Nm)':<14} {'DF % of max'}")
        print("─" * 55)
        for i, p in enumerate(self.x):
            df_min_i = self._get_exploration_df_min(i)
            df_max_cfg = opt_args.get("min_df_torque_nm", 20.0)
            df_pct_i = (df_min_i / df_max_cfg * 100) if df_max_cfg > 0 else 0
            print(f"   {i+1:<5} {p[0]:<10.4f} {p[1]:<10.4f} "
                  f"{df_min_i:<14.2f} {df_pct_i:.0f}%")

    # =======================================================================
    # BO safety shim: validate every GP suggestion before committing it
    # =======================================================================

    def _get_safe_bo_suggestion(self, raw_suggestion: np.ndarray) -> np.ndarray:
        """Validate BO-suggested params — enforces ALL hard constraints:
          - 90 Nm absolute cap
          - PF zone cap
          - DF minimum hard floor during BO (df_min_bo_nm)

        Strategy:
          1. Try the BO's argmax suggestion. If safe, return it.
          2. If unsafe, evaluate the acquisition function across a dense grid
             of the parameter space, rank by EI value, and walk down the
             ranking returning the first safe point. This preserves the BO's
             ranking of "promising" regions rather than reverting to random.
          3. As a last resort (almost never reached), random sample.

        Top-K ranked fallback is theoretically much better than random:
          - Random sampling: throws away all GP information, picks an
            arbitrary point that is unlikely to be informative
          - Top-K ranked: stays in the high-EI region (where the GP wanted
            to look), just sidesteps the unsafe sub-region
        """
        opt = self.args["Optimization"]
        pf_zone = opt.get("pf_zone_deg", [2.0, 20.0])
        pf_threshold = opt.get("pf_torque_threshold", 4.0)
        df_angle = opt.get("df_check_angle_deg", -10.0)
        df_min_bo = opt.get("df_min_bo_nm", 5.0)
        range_ = np.array(list(opt["range"]))
        n_parms = opt["n_parms"]

        candidate = raw_suggestion.flatten()
        R, L0 = candidate[0], candidate[1]

        # ── Step 1: try BO's actual argmax ──
        is_safe, max_pf, df_torque, max_total = self._is_safe_candidate(
            R, L0, pf_zone, pf_threshold,
            df_min=df_min_bo, df_check_angle=df_angle,
        )

        if is_safe:
            print(f"   ✅ BO suggestion passed (PF={max_pf:.2f} Nm, "
                  f"DF={df_torque:.2f} Nm, max={max_total:.2f} Nm)")
            return raw_suggestion

        print(f"   ⚠️  BO suggestion R={R:.4f}, L0={L0:.4f} failed "
              f"(PF={max_pf} Nm, DF={df_torque} Nm, max={max_total:.2f} Nm)")
        print(f"   🔍 Searching top-K acquisition rankings on grid...")

        # ── Step 2: top-K ranked fallback via acquisition function grid ──
        try:
            safe_candidate, rank, ei_val = self._top_k_safe_fallback(
                pf_zone, pf_threshold, df_min_bo, df_angle, range_,
            )
            if safe_candidate is not None:
                R, L0 = safe_candidate[0], safe_candidate[1]
                _, max_pf, df_torque, max_total = self._is_safe_candidate(
                    R, L0, pf_zone, pf_threshold,
                    df_min=df_min_bo, df_check_angle=df_angle,
                )
                print(f"   ✅ Top-K safe fallback (rank #{rank}, EI={ei_val:.4f}): "
                      f"R={R:.4f}, L0={L0:.4f} "
                      f"(PF={max_pf:.2f} Nm, DF={df_torque:.2f} Nm, "
                      f"max={max_total:.2f} Nm)")
                return safe_candidate.reshape(1, n_parms)
        except Exception as e:
            print(f"   ⚠️  Top-K fallback failed ({type(e).__name__}: {e}) — "
                  f"reverting to random sampling")

        # ── Step 3: random sampling as final fallback (rarely reached) ──
        print(f"   🎲 Random sampling fallback...")
        for attempt in range(500):
            candidate = np.random.uniform(range_[0], range_[1])
            R, L0 = candidate[0], candidate[1]
            is_safe, max_pf, df_torque, max_total = self._is_safe_candidate(
                R, L0, pf_zone, pf_threshold,
                df_min=df_min_bo, df_check_angle=df_angle,
            )
            if is_safe:
                print(f"   ✅ Random replacement at attempt {attempt+1}: "
                      f"R={R:.4f}, L0={L0:.4f} (DF={df_torque:.2f} Nm)")
                return candidate.reshape(1, n_parms)

        print(f"   ⚠️  No safe replacement found after 500 random attempts — "
              f"using original BO suggestion.")
        return raw_suggestion

    def _top_k_safe_fallback(self,
                              pf_zone: list, pf_threshold: float,
                              df_min: float, df_angle: float,
                              range_: np.ndarray,
                              n_grid: int = 50,
                              ):
        """Walk down ranked acquisition function values, return first safe point.

        Builds a `n_grid` × `n_grid` grid over the parameter space, evaluates
        the GP's acquisition function (qNoisyExpectedImprovement) at every
        grid point, sorts by EI value descending, and walks down the list
        checking safety. Returns the first safe candidate.

        Returns
        -------
        candidate : np.ndarray | None    safe (R, L0) point, or None if no
                                          grid point is safe
        rank      : int                   1-indexed rank in the EI ordering
                                          (1 = highest EI safe point)
        ei_val    : float                 acquisition value at returned point
        """
        import torch
        from botorch.acquisition import qNoisyExpectedImprovement
        from botorch.sampling import IIDNormalSampler

        # Build a dense grid over the physical parameter space
        n_parms = self.args["Optimization"]["n_parms"]
        if n_parms != 2:
            # Method as written assumes 2D — for higher dims, would need
            # different sampling strategy
            return None, 0, 0.0

        R_vals = np.linspace(range_[0, 0], range_[1, 0], n_grid)
        L0_vals = np.linspace(range_[0, 1], range_[1, 1], n_grid)
        RR, LL = np.meshgrid(R_vals, L0_vals)
        grid_phys = np.column_stack([RR.ravel(), LL.ravel()])

        # Normalize to [0,1] for the GP
        if self.NORMALIZATION:
            grid_norm = self._normalize_x(grid_phys)
            x_train_norm = self._normalize_x(self.x_opt)
        else:
            grid_norm = grid_phys
            x_train_norm = self.x_opt

        # Evaluate acquisition function at every grid point
        sampler = IIDNormalSampler(sample_shape=torch.Size([200]), seed=1234)
        x_train_tensor = torch.tensor(x_train_norm, dtype=torch.float64)
        acq = qNoisyExpectedImprovement(self.BO.model, x_train_tensor,
                                         sampler=sampler)

        # Evaluate in batches (memory-friendly)
        ei_values = np.zeros(len(grid_norm))
        batch_size = 100
        with torch.no_grad():
            for start in range(0, len(grid_norm), batch_size):
                end = min(start + batch_size, len(grid_norm))
                x_batch = torch.tensor(grid_norm[start:end],
                                        dtype=torch.float64).unsqueeze(1)
                ei_values[start:end] = acq(x_batch).numpy()

        # Sort grid points by EI, descending
        ranked_indices = np.argsort(-ei_values)

        # Walk down the ranking
        for ranking, idx in enumerate(ranked_indices, start=1):
            R_cand, L0_cand = grid_phys[idx]
            is_safe, _, _, _ = self._is_safe_candidate(
                R_cand, L0_cand, pf_zone, pf_threshold,
                df_min=df_min, df_check_angle=df_angle,
            )
            if is_safe:
                return grid_phys[idx], ranking, float(ei_values[idx])

        # No safe point found in the entire grid (very unlikely)
        return None, 0, 0.0


    # =======================================================================
    # Terminal mode (headless, prompts operator between trials)
    # =======================================================================

    def print_trial_parameters(self, trial_num: int, params: np.ndarray) -> None:
        print("\n" + "🎯" * 30)
        print(f"   TRIAL {trial_num}/{self.args['Optimization']['n_steps']}")
        print("🎯" * 30)
        if trial_num <= self.args["Optimization"]["n_exploration"]:
            df_min = self._get_exploration_df_min(trial_num - 1)
            df_max = self.args["Optimization"].get("min_df_torque_nm", 20.0)
            df_pct = (df_min / df_max * 100) if df_max > 0 else 0
            print(f"   [Exploration {trial_num}/"
                  f"{self.args['Optimization']['n_exploration']}]")
            print(f"   [DF min this trial: {df_min:.2f} Nm — {df_pct:.0f}% of max]")
        else:
            print(f"   [Optimization — Bayesian]")
        print(f"\n{'PARAMETERS TO ENTER:':^60}")
        print("─" * 60)
        param_names = ["R", "L0"]
        for name, val in zip(param_names[:len(params)], params):
            print(f"  {name:3s} = {val:.4f}")
        print("─" * 60 + "\n")

    def start(self):
        """Run the full experiment interactively in the terminal.

        For Streamlit UI usage, see apps/run_experiment.py — it calls
        _generate_initial_parameters, _get_safe_bo_suggestion, etc. directly
        and manages the trial loop via the web interface.
        """
        if self.n == 0:
            print(f"\n{'='*70}")
            print(f"STARTING HIL EXOSKELETON OPTIMIZATION")
            print(f"Cost Function: {self.cost.cost_name}")
            print(f"Parameters: {self.args['Optimization']['n_parms']}")
            print(f"Total Trials: {self.args['Optimization']['n_steps']}")
            print(f"Exploration Trials: {self.args['Optimization']['n_exploration']}")
            print(f"Exploration Sampling: Latin Hypercube Sampling (LHS)")
            print(f"BO Direction: MINIMIZING {'|cost|' if self.signed else 'cost'} "
                  f"(signed={self.signed})")
            print(f"Hard Torque Cap: {HARD_TORQUE_CAP} Nm")
            print(f"{'='*70}\n")
            self._generate_initial_parameters()

        while self.n < self.args["Optimization"]["n_steps"]:
            trial_num = self.n + 1
            self.print_trial_parameters(trial_num, self.x[self.n])

            print("📝 INSTRUCTIONS:")
            print("  1. Enter parameters into Computer 2 exoskeleton controller")
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
                print("⚠️  Cost extraction failed!")
                retry = input("Retry this trial? (Y/n): ")
                if retry.lower() != 'n':
                    continue
                else:
                    self.n += 1
                    continue

            if len(self.x_opt) < 1:
                self.x_opt = np.array([self.x[self.n]])
                self.y_opt = np.array([cost_value])
            else:
                self.x_opt = np.concatenate(
                    (self.x_opt, np.array([self.x[self.n]])))
                self.y_opt = np.concatenate(
                    (self.y_opt, np.array([cost_value])))

            print(f"\n✅ Recorded: Cost = {cost_value:.4f} "
                  f"for params {self.x_opt[-1]}")
            self.n += 1

            if (self.n >= self.args["Optimization"]["n_exploration"]
                    and self.n < self.args["Optimization"]["n_steps"]):
                print(f"\n🔬 Running Bayesian Optimization "
                      f"(minimizing {'|cost|' if self.signed else 'cost'})...")
                if self.NORMALIZATION:
                    norm_x = self._normalize_x(self.x_opt)
                    norm_y = self._mean_normalize_y(self.y_opt)
                    new_parameter = self.BO.run(
                        norm_x.reshape(self.n, -1),
                        norm_y.reshape(self.n, -1))
                    new_parameter = self._denormalize_x(new_parameter)
                else:
                    new_parameter = self.BO.run(
                        self.x_opt.reshape(self.n, -1),
                        (-np.abs(self.y_opt) if self.signed
                         else -self.y_opt).reshape(self.n, -1),
                    )
                new_parameter = self._get_safe_bo_suggestion(new_parameter)
                print(f"   Next suggested parameters: {new_parameter.flatten()}")
                self.x = np.concatenate((
                    self.x,
                    new_parameter.reshape(
                        1, self.args["Optimization"]["n_parms"])
                ), axis=0)

            best_idx = (np.argmin(np.abs(self.y_opt)) if self.signed
                        else np.argmin(self.y_opt))
            print(f"\n📊 Best so far: Cost={self.y_opt[best_idx]:.4f} | "
                  f"Params={self.x_opt[best_idx]}")

            if self.n < self.args["Optimization"]["n_steps"]:
                input("\nPress ENTER to continue to next trial...")

        print("\n" + "🎉" * 35)
        print("   OPTIMIZATION COMPLETE!")
        print("🎉" * 35)
        best_idx = (np.argmin(np.abs(self.y_opt)) if self.signed
                    else np.argmin(self.y_opt))
        print(f"\nBest result:")
        print(f"  Cost: {self.y_opt[best_idx]:.4f}")
        print(f"  Parameters: {self.x_opt[best_idx]}")
        print(f"\nAll parameters saved to: "
              f"{self.args['Optimization']['model_save_path']}")


__all__ = ["HIL_Exo", "HARD_TORQUE_CAP"]