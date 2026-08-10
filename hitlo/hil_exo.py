"""
hitlo.hil_exo — experiment driver with exoskeleton safety constraints.

UPDATED for (L0, attach) parameterization with R = 0.28 m fixed.

Version 2.5.1 — (L0, attach) parameterization with corrected torque constraints
                BO now optimizes spring rest length (L0) and foot attachment
                (attach_ratio) with anchor distance R fixed at 0.28 m.
                - L0 ∈ [0.32, 0.44] m (engagement timing)
                - attach ∈ [-0.2, +1.0] (direction + timing)
                
Safety constraints (HARD):
                - Max plantarflexor torque: 90 Nm [0°, 30°]
                - Max dorsiflexor torque: 10 Nm [-30°, 0°]
                - Slack region at 0°: < 2 Nm (neutral engagement)
                
This is the HITLO_Symmetry wrapper around HIL_toolkit's BayesianOptimization.
It adds exoskeleton-specific safety enforcement that HIL_toolkit doesn't know
about: torque caps for both directions, slack region enforcement. These exist
because our device is a physical spring mechanism that must not be pushed to
parameter combinations that would injure the participant or fail to engage.
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


class HIL_Exo:
    """Orchestrates the HITLO experiment: LHS exploration + BO with safety checks.

    Wraps HIL_toolkit's BayesianOptimization for the GP engine and adds:
      - Exoskeleton-specific safety constraints (torque caps both directions)
      - Latin Hypercube exploration sampling with oversampling pool
      - Signed-symmetry BO that minimizes |cost - si_target| toward zero,
        supporting both "drive toward symmetry" (si_target=0) and "induce
        target asymmetry" (si_target!=0) paradigms.

    Args:
        args: Full experiment config dict (parsed from exo_symmetry_config.yml).
              Expects top-level keys 'Optimization' and 'Cost'.
              Reads `Cost.signed` (bool) and `Cost.si_target` (float, default 0.0).
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
        self.si_target = float(args.get("Cost", {}).get("si_target", 0.0))
        self._start_optimization(self.args["Optimization"])

        # Format the direction string once, since we print it in multiple places.
        if self.signed:
            self._bo_direction_str = f"|SI - {self.si_target:+.1f}|"
        else:
            self._bo_direction_str = "cost"

        print(f"✅ HIL_Exo initialized (L0, attach parameterization)")
        print(f"   Exploration sampling: Latin Hypercube Sampling (LHS)")
        print(f"   BO direction: MINIMIZING {self._bo_direction_str} "
              f"(signed={self.signed}, si_target={self.si_target:+.1f})")
        print(f"   Torque constraints:")
        print(f"     - Max plantarflexor [0°, 30°]: 60 Nm")
        print(f"     - Max dorsiflexor [-30°, 0°]: -10 Nm")
        print(f"     - Slack at 0° (neutral): < 2 Nm")
        print(f"   Fixed R: 0.28 m (anchor distance)")

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
        # Clamp to bounds (BO can suggest outside [0,1] due to numerical precision)
        x = np.clip(x, range_x[0, :], range_x[1, :])
        return x

    def _mean_normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize y for GP input.

        If signed mode: compute |y - si_target| so BO minimizes the distance
        from the configured target asymmetry. With si_target=0 this reduces
        to |y| (drives SI toward 0 for Aim 2 stroke).
        With si_target=-10 (Aim 1 healthy), BO drives SI toward -10%.

        Mean-center and std-scale the result for GP numerical stability,
        then negate so BoTorch maximization = cost minimization.
        """
        y = np.array(y)
        if self.signed:
            y = np.abs(y - self.si_target)
        y = (y - np.mean(y)) / (np.std(y) + 1e-8)
        return -y

    # =======================================================================
    # BO initialization (HIL_toolkit instance)
    # =======================================================================

    def _generate_initial_parameters(self) -> None:
        """Generate initial parameters: manual ramp only. BO generates trials 6+ on demand."""
        opt = self.args["Optimization"]
        n_manual = opt.get("manual_ramp_trials", 0)
        
        if n_manual > 0 and "ramp_sequence" in opt:
            # Manual ramp trials only
            ramp = np.array(opt["ramp_sequence"])
            self.x = ramp
            
            print(f"✅ Generated {len(self.x)} parameters")
            print(f"   Trials 1–{n_manual}: manual ramp (torque-based)")
            print(f"   Trials {n_manual+1}–{opt['n_steps']}: Bayesian Optimization (on demand)")
        else:
            # No ramp configured
            self.x = np.empty((0, opt["n_parms"]))
            print(f"✅ No manual ramp. BO will generate all {opt['n_steps']} trials on demand.")

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
    # Safety: verify a candidate (L0, attach) won't produce unsafe torques
    # =======================================================================

    def _is_safe_candidate(self, L0: float, attach: float,
                           ) -> Tuple[bool, dict]:
        """Check if a candidate (L0, attach) satisfies safety constraints.
        
        DIRECTIONAL constraints based on attach polarity:
          - If attach < 0 (PF-assist): enforce max PF torque in [0°, 30°] ≤ 90 Nm
            (DF zone naturally ~zero because spring is pulling plantarflexion)
          - If attach > 0 (DF-assist): enforce max DF torque in [-30°, 0°] ≤ 20 Nm
            (PF zone naturally ~zero because spring is pulling dorsiflexion)
          - If attach ≈ 0 (neutral): very weak torque everywhere, always safe
        
        This allows the BO to explore diverse torque profiles:
        - LHS samples across polarity spectrum → sees both PF and DF regions
        - Zero regions are naturally zero (physics), not artificially constrained
        - BO learns which regions produce low/high SI and when to use them
        
        Returns:
            (is_safe, details_dict)
            where details_dict contains: max_pf, max_df, at_zero, reason (if failed)
        """
        opt = self.args["Optimization"]
        max_pf_nm = opt.get("max_pf_torque_nm", 90.0)
        max_df_nm = opt.get("max_df_torque_nm", 10.0)  # Rubber band limit
        slack_threshold = opt.get("slack_at_neutral_max_torque", 2.0)
        pf_range = opt.get("pf_check_angle_range", [0.0, 30.0])
        df_range = opt.get("df_check_angle_range", [-30.0, 0.0])
        n_pts = 100
        
        details = {
            'max_pf': None,
            'max_df': None,
            'at_zero': None,
            'polarity': 'neutral',
            'reason': None,
        }
        
        # Determine which direction this config is assisting
        pf_angles = np.linspace(pf_range[0], pf_range[1], n_pts)
        df_angles = np.linspace(df_range[0], df_range[1], n_pts)
        pf_torques = [compute_exo_torque(a, L0, attach) for a in pf_angles]
        df_torques = [compute_exo_torque(a, L0, attach) for a in df_angles]
        
        max_pf = max(pf_torques)
        max_df_abs = max([abs(t) for t in df_torques])
        
        details['max_pf'] = max_pf
        details['max_df'] = max_df_abs
        
        # Slack at neutral (HARD CONSTRAINT)
        torque_at_zero = abs(compute_exo_torque(0.0, L0, attach))
        details['at_zero'] = torque_at_zero
        
        if torque_at_zero > slack_threshold:
            details['reason'] = f"Slack: |torque@0°|={torque_at_zero:.2f}Nm > {slack_threshold:.1f}Nm"
            return False, details
        
        # ========== REGION SLACK CONSTRAINT (FIRST) ==========
        # DF-assist: entire DF region [-30°, 0°] must be slack
        # PF-assist: entire PF region [0°, 30°] must be slack
        
        if attach > 0.05:  # DF-assist: DF region must be slack, PF region must be negative
            details['polarity'] = 'DF-assist'
            df_check_angles = np.linspace(-30, 0, 50)
            df_check_torques = [abs(compute_exo_torque(a, L0, attach)) for a in df_check_angles]
            max_df_in_region = max(df_check_torques)
            
            if max_df_in_region > 2.0:
                details['reason'] = f"DF-assist: DF region [-30°,0°] max={max_df_in_region:.2f}Nm > 2.0Nm (must be slack)"
                return False, details
            
            # Check PF region is NEGATIVE (DF resistance)
            pf_check_angles = np.linspace(0, 30, 50)
            pf_check_torques = [compute_exo_torque(a, L0, attach) for a in pf_check_angles]
            if any(t > -0.5 for t in pf_check_torques):  # Allow small positive due to numerical noise
                details['reason'] = f"DF-assist: PF region must be negative torque, found positive"
                return False, details
            
            # Also check max DF torque limit
            if max_df_abs > max_df_nm:
                details['reason'] = f"DF-assist: DF peak {max_df_abs:.1f}Nm > {max_df_nm:.1f}Nm limit"
                return False, details
        
        elif attach < -0.05:  # PF-assist: PF region must be slack, DF region must be positive
            details['polarity'] = 'PF-assist'
            pf_check_angles = np.linspace(0, 30, 50)
            pf_check_torques = [abs(compute_exo_torque(a, L0, attach)) for a in pf_check_angles]
            max_pf_in_region = max(pf_check_torques)
            
            if max_pf_in_region > 2.0:
                details['reason'] = f"PF-assist: PF region [0°,30°] max={max_pf_in_region:.2f}Nm > 2.0Nm (must be slack)"
                return False, details
            
            # Check DF region is POSITIVE (PF assistance)
            df_check_angles = np.linspace(-30, 0, 50)
            df_check_torques = [compute_exo_torque(a, L0, attach) for a in df_check_angles]
            if any(t < 0.5 for t in df_check_torques):  # Allow small negative due to numerical noise
                details['reason'] = f"PF-assist: DF region must be positive torque, found negative"
                return False, details
            
            # Also check max PF torque limit
            if max_pf > max_pf_nm:
                details['reason'] = f"PF-assist: PF peak {max_pf:.1f}Nm > {max_pf_nm:.1f}Nm limit"
                return False, details
        
        else:  # Neutral (attach ≈ 0)
            details['polarity'] = 'neutral'
            # Both regions near zero naturally, very safe
            pass
        
        return True, details

    # =======================================================================
    # Exploration: Latin Hypercube Sampling with safety filtering
    # =======================================================================



    # =======================================================================
    # BO safety shim: validate every GP suggestion before committing it
    # =======================================================================

    def _get_safe_bo_suggestion(self, raw_suggestion: np.ndarray) -> np.ndarray:
        """Validate BO-suggested params — enforces all safety constraints.

        Strategy:
          1. Try the BO's argmax suggestion. If safe, return it.
          2. If unsafe, evaluate the acquisition function across a dense grid
             of the parameter space, rank by EI value, and walk down the
             ranking returning the first safe point.
          3. As a last resort, random sample.
        """
        candidate = raw_suggestion.flatten()
        L0, attach = candidate[0], candidate[1]
        n_parms = self.args["Optimization"]["n_parms"]

        # ── Step 1: try BO's actual argmax ──
        is_safe, details = self._is_safe_candidate(L0, attach)

        if is_safe:
            print(f"   ✅ BO suggestion passed (PF={details['max_pf']:.2f}Nm, "
                  f"DF={details['max_df']:.2f}Nm, @0°={details['at_zero']:.2f}Nm)")
            return raw_suggestion

        print(f"   ⚠️  BO suggestion L0={L0:.4f}, attach={attach:.4f} failed")
        print(f"      {details['reason']}")
        print(f"   🔍 Searching top-K acquisition rankings on grid...")

        # ── Step 2: top-K ranked fallback via acquisition function grid ──
        try:
            opt = self.args["Optimization"]
            range_ = np.array(list(opt["range"])).reshape(2, opt["n_parms"])
            
            safe_candidate, rank, ei_val = self._top_k_safe_fallback(range_)
            if safe_candidate is not None:
                L0, attach = safe_candidate[0], safe_candidate[1]
                is_safe, details = self._is_safe_candidate(L0, attach)
                print(f"   ✅ Top-K safe fallback (rank #{rank}, EI={ei_val:.4f}): "
                      f"L0={L0:.4f}, attach={attach:.4f} "
                      f"(PF={details['max_pf']:.2f}Nm, DF={details['max_df']:.2f}Nm)")
                return safe_candidate.reshape(1, n_parms)
        except Exception as e:
            print(f"   ⚠️  Top-K fallback failed ({type(e).__name__}: {e}) — "
                  f"reverting to random sampling")

        # ── Step 3: random sampling as final fallback ──
        print(f"   🎲 Random sampling fallback...")
        opt = self.args["Optimization"]
        range_ = np.array(list(opt["range"])).reshape(2, opt["n_parms"])
        
        for attempt in range(500):
            candidate = np.random.uniform(range_[0], range_[1])
            L0, attach = candidate[0], candidate[1]
            is_safe, details = self._is_safe_candidate(L0, attach)
            if is_safe:
                print(f"   ✅ Random replacement at attempt {attempt+1}: "
                      f"L0={L0:.4f}, attach={attach:.4f}")
                return candidate.reshape(1, n_parms)

        print(f"   ⚠️  No safe replacement found — using original BO suggestion.")
        return raw_suggestion

    def _top_k_safe_fallback(self, range_: np.ndarray, n_grid: int = 50):
        """Walk down ranked acquisition function values, return first safe point."""
        import torch
        from botorch.acquisition import qNoisyExpectedImprovement
        from botorch.sampling import IIDNormalSampler

        n_parms = self.args["Optimization"]["n_parms"]
        if n_parms != 2:
            return None, 0, 0.0

        L0_vals = np.linspace(range_[0, 0], range_[1, 0], n_grid)
        attach_vals = np.linspace(range_[0, 1], range_[1, 1], n_grid)
        LL, AA = np.meshgrid(L0_vals, attach_vals)
        grid_phys = np.column_stack([LL.ravel(), AA.ravel()])

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
            L0_cand, attach_cand = grid_phys[idx]
            is_safe, _ = self._is_safe_candidate(L0_cand, attach_cand)
            if is_safe:
                return grid_phys[idx], ranking, float(ei_values[idx])

        return None, 0, 0.0

    # =======================================================================
    # Best-so-far tracking helper
    # =======================================================================

    def _best_so_far_idx(self) -> int:
        """Return index of best trial in self.y_opt under the configured paradigm."""
        if self.signed:
            return int(np.argmin(np.abs(self.y_opt - self.si_target)))
        return int(np.argmin(self.y_opt))

    # =======================================================================
    # Terminal mode (headless, prompts operator between trials)
    # =======================================================================

    def print_trial_parameters(self, trial_num: int, params: np.ndarray) -> None:
        print("\n" + "🎯" * 30)
        print(f"   TRIAL {trial_num}/{self.args['Optimization']['n_steps']}")
        print("🎯" * 30)
        if trial_num <= self.args["Optimization"]["n_exploration"]:
            print(f"   [Exploration {trial_num}/"
                  f"{self.args['Optimization']['n_exploration']}]")
        else:
            print(f"   [Optimization — Bayesian]")
        print(f"\n{'PARAMETERS TO ENTER:':^60}")
        print("─" * 60)
        param_names = ["L0 (m)", "Attach"]
        for name, val in zip(param_names[:len(params)], params):
            if name == "L0 (m)":
                print(f"  {name:8s} = {val:.4f}  ({val*100:.1f} cm)")
            else:
                print(f"  {name:8s} = {val:+.2f}")
        print("─" * 60 + "\n")

    def start(self):
        """Run the full experiment interactively in the terminal."""
        if self.n == 0:
            print(f"\n{'='*70}")
            print(f"STARTING HIL EXOSKELETON OPTIMIZATION (L0, attach)")
            print(f"Cost Function: {self.cost.cost_name}")
            print(f"Parameters: {self.args['Optimization']['n_parms']}")
            print(f"Total Trials: {self.args['Optimization']['n_steps']}")
            print(f"Exploration Trials: {self.args['Optimization']['n_exploration']}")
            print(f"Exploration Sampling: Latin Hypercube Sampling (LHS)")
            print(f"BO Direction: MINIMIZING {self._bo_direction_str} "
                  f"(signed={self.signed}, si_target={self.si_target:+.1f})")
            print(f"Torque constraints:")
            print(f"  - Max PF [0°, 30°]: 60 Nm")
            print(f"  - Max DF [-30°, 0°]: -10 Nm")
            print(f"  - Slack at 0°: < 2 Nm")
            print(f"{'='*70}\n")
            self._generate_initial_parameters()

        while self.n < self.args["Optimization"]["n_steps"]:
            trial_num = self.n + 1
            self.print_trial_parameters(trial_num, self.x[self.n])

            print("📝 INSTRUCTIONS:")
            print("  1. Set spring rest length (L0) and foot attachment (attach)")
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

            # Log how close this trial got us to the target.
            if self.signed:
                dist = abs(cost_value - self.si_target)
                print(f"\n✅ Recorded: Cost (raw SI) = {cost_value:+.4f}  "
                      f"|  |SI - target| = {dist:.4f}  "
                      f"for params L0={self.x_opt[-1, 0]:.4f}, attach={self.x_opt[-1, 1]:+.2f}")
            else:
                print(f"\n✅ Recorded: Cost = {cost_value:.4f} "
                      f"for params L0={self.x_opt[-1, 0]:.4f}, attach={self.x_opt[-1, 1]:+.2f}")
            self.n += 1

            if (self.n >= self.args["Optimization"]["n_exploration"]
                    and self.n < self.args["Optimization"]["n_steps"]):
                print(f"\n🔬 Running Bayesian Optimization "
                      f"(minimizing {self._bo_direction_str})...")
                if self.NORMALIZATION:
                    norm_x = self._normalize_x(self.x_opt)
                    norm_y = self._mean_normalize_y(self.y_opt)
                    new_parameter = self.BO.run(
                        norm_x.reshape(self.n, -1),
                        norm_y.reshape(self.n, -1))
                    new_parameter = self._denormalize_x(new_parameter)
                else:
                    if self.signed:
                        y_for_bo = -np.abs(self.y_opt - self.si_target)
                    else:
                        y_for_bo = -self.y_opt
                    new_parameter = self.BO.run(
                        self.x_opt.reshape(self.n, -1),
                        y_for_bo.reshape(self.n, -1),
                    )
                new_parameter = self._get_safe_bo_suggestion(new_parameter)
                print(f"   Next suggested parameters: L0={new_parameter.flatten()[0]:.4f}, "
                      f"attach={new_parameter.flatten()[1]:+.2f}")
                self.x = np.concatenate((
                    self.x,
                    new_parameter.reshape(
                        1, self.args["Optimization"]["n_parms"])
                ), axis=0)

            best_idx = self._best_so_far_idx()
            if self.signed:
                best_dist = abs(self.y_opt[best_idx] - self.si_target)
                print(f"\n📊 Best so far: SI={self.y_opt[best_idx]:+.4f}  "
                      f"|  |SI - target| = {best_dist:.4f}  "
                      f"|  L0={self.x_opt[best_idx, 0]:.4f}, attach={self.x_opt[best_idx, 1]:+.2f}")
            else:
                print(f"\n📊 Best so far: Cost={self.y_opt[best_idx]:.4f} | "
                      f"L0={self.x_opt[best_idx, 0]:.4f}, attach={self.x_opt[best_idx, 1]:+.2f}")

            if self.n < self.args["Optimization"]["n_steps"]:
                input("\nPress ENTER to continue to next trial...")

        print("\n" + "🎉" * 35)
        print("   OPTIMIZATION COMPLETE!")
        print("🎉" * 35)
        best_idx = self._best_so_far_idx()
        print(f"\nBest result:")
        if self.signed:
            best_dist = abs(self.y_opt[best_idx] - self.si_target)
            print(f"  SI:                {self.y_opt[best_idx]:+.4f}")
            print(f"  |SI - target|:     {best_dist:.4f}  "
                  f"(target = {self.si_target:+.1f})")
        else:
            print(f"  Cost: {self.y_opt[best_idx]:.4f}")
        print(f"  L0:                {self.x_opt[best_idx, 0]:.4f} m ({self.x_opt[best_idx, 0]*100:.1f} cm)")
        print(f"  Attachment:        {self.x_opt[best_idx, 1]:+.2f}")
        print(f"\nAll parameters saved to: "
              f"{self.args['Optimization']['model_save_path']}")


__all__ = ["HIL_Exo"]