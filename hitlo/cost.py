"""
hitlo.cost — BO cost function: step-time symmetry + exoskeleton spring penalty.

This module is the glue between the detection pipeline (hitlo.detection),
the symmetry metric (hitlo.symmetry), and the Bayesian optimization loop.

The `SymmetryCost` class is the object that HIL_Exo calls each trial to
evaluate the current (R, L0) parameter suggestion.

Version 2.0.0 (refactored from symmetry_cost.py v1.8.0):
  - Detection logic moved to hitlo.detection (shared with diagnostic tool)
  - Symmetry logic moved to hitlo.symmetry
  - I/O logic moved to hitlo.io
  - This file contains only the BO-specific glue: the class wrapper,
    spring-torque penalty math, and the per-trial orchestration.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import os
import numpy as np

from hitlo.detection import DetectionConfig, detect_heelstrikes_full
from hitlo.symmetry import (
    compute_step_times, compute_symmetry_index,
    trim_peaks, filter_implausible_strides,
)
from hitlo.io import load_both_polar_streams, load_polar_stream, trial_filename


# ===========================================================================
# Spring-torque model (physics of the passive exo)
# ===========================================================================

def compute_exo_torque(ankle_angle_deg: float, R: float, L0: float) -> float:
    """Ankle torque produced by the passive spring at a given ankle angle.

    Physical model of the LegExoNET spring-pulley mechanism: given anchor
    position R (m from ankle pivot) and resting length L0 (m), compute the
    torque the stretched spring applies to the ankle at a given angle.

    Conventions: positive ankle angle = plantarflexion (PF), negative = DF.
    Positive torque = dorsiflexion assist (what we want during swing).
    """
    k = 12000.0
    segment_length = 0.335
    theta = 196.0
    attachment_ratio = -0.2

    ankle_x, ankle_y = 0.0, 0.0

    angle_rad = np.radians(-ankle_angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R_ankle = np.array([[cos_a, -sin_a],
                        [sin_a, cos_a]])

    heel_rel = np.array([-0.08, -0.05])
    toe_rel = np.array([segment_length - 0.08, -0.05])

    rotated_heel = R_ankle @ heel_rel + np.array([ankle_x, ankle_y])
    rotated_toe = R_ankle @ toe_rel + np.array([ankle_x, ankle_y])

    attach_x = rotated_heel[0] + attachment_ratio * (rotated_toe[0] - rotated_heel[0])
    attach_y = rotated_heel[1] + attachment_ratio * (rotated_toe[1] - rotated_heel[1])

    anchor_angle = theta - 90.0
    anchor_x = ankle_x + R * np.cos(np.radians(anchor_angle))
    anchor_y = ankle_y + R * np.sin(np.radians(anchor_angle))

    Ldist = np.sqrt((attach_x - anchor_x) ** 2 + (attach_y - anchor_y) ** 2)
    if np.isnan(Ldist) or np.isinf(Ldist) or Ldist <= 1e-6:
        return 0.0

    tension = k * max(Ldist - L0, 0.0)
    force_x = tension * (anchor_x - attach_x) / Ldist
    force_y = tension * (anchor_y - attach_y) / Ldist
    if np.isnan(force_x) or np.isnan(force_y) or np.isinf(force_x) or np.isinf(force_y):
        return 0.0

    lever_x = attach_x - ankle_x
    lever_y = attach_y - ankle_y
    taudes = -(lever_x * force_y - lever_y * force_x)
    if np.isnan(taudes) or np.isinf(taudes):
        return 0.0
    return float(taudes)


def compute_torque_curve(R: float, L0: float,
                         angle_min: float = -30.0,
                         angle_max: float = 30.0,
                         n_points: int = 200
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the torque model across a range of ankle angles (for plotting)."""
    angles = np.linspace(angle_min, angle_max, n_points)
    torques = np.array([compute_exo_torque(a, R, L0) for a in angles])
    return angles, torques


def compute_spring_penalty(R: float, L0: float,
                           pf_zone: Tuple[float, float] = (2.0, 20.0),
                           df_angle: float = -10.0,
                           lambda_pf: float = 1.0,
                           mu_df: float = 0.5,
                           n_points: int = 200) -> float:
    """Shape penalty added to the symmetry cost.

    Penalizes:
      - large torques in the plantarflexion zone (should be ≈ 0 during PF)
      - insufficient dorsiflexion assist at df_angle (we want positive torque)
    """
    pf_angles = np.linspace(pf_zone[0], pf_zone[1], n_points)
    pf_torques = np.array([compute_exo_torque(a, R, L0) for a in pf_angles])
    pf_penalty = float(np.mean(pf_torques ** 2))

    df_torque = compute_exo_torque(df_angle, R, L0)
    df_reward = max(df_torque, 0.0)

    return lambda_pf * pf_penalty - mu_df * df_reward


# ===========================================================================
# Per-trial analysis result (returned for inspection / QC / logging)
# ===========================================================================

@dataclass
class TrialAnalysis:
    """Everything computed for one trial. BO uses `total_cost`; QC uses the rest."""
    # Headline numbers
    total_cost: float
    symmetry_index: float
    spring_penalty: float

    # Step times
    right_step_times: np.ndarray
    left_step_times: np.ndarray
    per_stride_si: np.ndarray

    # Heel strike times (post-trim, post-plausibility)
    left_heel_strikes: np.ndarray
    right_heel_strikes: np.ndarray

    # Raw detection results (for QC plotting)
    left_result: Optional[object] = None    # DetectionResult
    right_result: Optional[object] = None   # DetectionResult

    # Warnings collected during analysis
    warnings: List[str] = field(default_factory=list)


# ===========================================================================
# Cost class
# ===========================================================================

class SymmetryCost:
    """BO cost extractor for the HITLO exoskeleton optimization loop.

    Usage
    -----
    >>> cost = SymmetryCost(trial_data_dir="/path/to/xdf/files",
    ...                     subject_id="P048", session="S001",
    ...                     signed=True, trim_seconds=3.0)
    >>> cost.set_params(R=0.08, L0=0.25)
    >>> total_cost = cost.extract_cost_from_file(trial_num=1)
    """

    def __init__(self,
                 trial_data_dir: str,
                 subject_id: str = "",
                 session: str = "",
                 detection_cfg: Optional[DetectionConfig] = None,
                 # spring penalty
                 lambda_pf: float = 0.01,
                 mu_df: float = 0.005,
                 pf_zone: Tuple[float, float] = (2.0, 20.0),
                 df_angle: float = -10.0,
                 # symmetry behavior
                 signed: bool = False,
                 trim_seconds: float = 3.0,
                 ):
        self.cost_name = "gait_symmetry"
        self.trial_data_dir = trial_data_dir
        self.subject_id = subject_id
        self.session = session

        self.detection_cfg = detection_cfg or DetectionConfig()

        self.lambda_pf = lambda_pf
        self.mu_df = mu_df
        self.pf_zone = pf_zone
        self.df_angle = df_angle
        self.signed = signed
        self.trim_seconds = trim_seconds

        self._R: Optional[float] = None
        self._L0: Optional[float] = None

        print(f"✅ SymmetryCost v2.0.0 initialized")
        print(f"   Mode:              {'SIGNED' if signed else 'UNSIGNED (abs)'}")
        print(f"   Directory:         {trial_data_dir}")
        print(f"   Detection:         cluster-keep-last + stance confirm")
        print(f"   Jerk threshold:    {self.detection_cfg.strict_thresh:.2f} SD strict / "
              f"{self.detection_cfg.recovery_thresh:.2f} SD recovery")
        print(f"   Cluster gap:       {self.detection_cfg.cluster_gap_s}s")
        print(f"   Stance check:      {self.detection_cfg.stance_duration_s}s window, "
              f"±{self.detection_cfg.stance_tolerance_pct*100:.0f}% of baseline")
        if trim_seconds > 0:
            print(f"   Steady-state trim: {trim_seconds:.1f}s from each end")

    def set_params(self, R: float, L0: float) -> None:
        """Set spring parameters for the current trial (for penalty calculation)."""
        self._R = R
        self._L0 = L0

    # ----- main entry points ---------------------------------------------

    def extract_cost_from_file(self,
                               trial_num: int,
                               filename: Optional[str] = None
                               ) -> Optional[float]:
        """Analyze one trial and return the total cost (or np.nan on failure).

        This is the method HIL_Exo calls. For richer access (heel strikes,
        per-stride SI, etc.) use `analyze_trial()` directly.
        """
        analysis = self.analyze_trial(trial_num=trial_num, filename=filename)
        if analysis is None:
            return None
        return analysis.total_cost

    def analyze_trial(self,
                      trial_num: int,
                      filename: Optional[str] = None,
                      verbose: bool = True,
                      ) -> Optional[TrialAnalysis]:
        """Full analysis of one trial; returns everything needed for QC + BO."""
        if filename is None:
            filename = trial_filename(self.subject_id, self.session, trial_num)
        xdf_path = os.path.join(self.trial_data_dir, filename)

        if verbose:
            print("\n" + "=" * 60)
            print(f"ANALYZING TRIAL {trial_num}")
            print("=" * 60)
            print(f"📂 Loading {xdf_path}...")

        left, right = load_both_polar_streams(xdf_path)

        if left is None or right is None:
            if verbose:
                print("⚠️  Two-sensor streams not found — falling back to single sensor")
            return self._analyze_single_sensor(xdf_path, verbose=verbose)

        if verbose:
            print(f"✅ Left:  {len(left.accel)} samples, "
                  f"{left.actual_fs:.2f} Hz")
            print(f"✅ Right: {len(right.accel)} samples, "
                  f"{right.actual_fs:.2f} Hz")

        warnings: List[str] = []

        # --- Detection (uses hitlo.detection pipeline) ---
        left_result = detect_heelstrikes_full(left.accel, left.timestamps,
                                              cfg=self.detection_cfg)
        right_result = detect_heelstrikes_full(right.accel, right.timestamps,
                                               cfg=self.detection_cfg)

        if verbose:
            print(f"   Left:  {len(left_result.strict_peaks)} strict "
                  f"+ {len(left_result.recovered_peaks)} recovered "
                  f"= {len(left_result.all_candidates)} candidates "
                  f"→ {len(left_result.heel_strike_indices)} heel strikes")
            print(f"   Right: {len(right_result.strict_peaks)} strict "
                  f"+ {len(right_result.recovered_peaks)} recovered "
                  f"= {len(right_result.all_candidates)} candidates "
                  f"→ {len(right_result.heel_strike_indices)} heel strikes")

        left_times_raw = left_result.heel_strike_times
        right_times_raw = right_result.heel_strike_times

        if len(left_times_raw) < 3 or len(right_times_raw) < 3:
            if verbose:
                print("⚠️  Not enough heel strikes detected!")
            return None

        # --- Trim + plausibility filter ---
        trial_start = min(left.timestamps[0], right.timestamps[0])
        trial_end = max(left.timestamps[-1], right.timestamps[-1])
        left_times = trim_peaks(left_times_raw, trial_start, trial_end, self.trim_seconds)
        right_times = trim_peaks(right_times_raw, trial_start, trial_end, self.trim_seconds)

        if self.trim_seconds > 0 and verbose:
            print(f"   Trimmed ±{self.trim_seconds}s: "
                  f"Left {len(left_times_raw)}→{len(left_times)}, "
                  f"Right {len(right_times_raw)}→{len(right_times)}")

        if len(left_times) < 3 or len(right_times) < 3:
            if verbose:
                print("⚠️  Not enough peaks after trim!")
            return None

        left_times, l_fast, l_slow = filter_implausible_strides(left_times)
        right_times, r_fast, r_slow = filter_implausible_strides(right_times)
        if l_fast + l_slow > 0:
            warnings.append(
                f"LEFT: {l_fast} implausibly-fast, {l_slow} implausibly-slow strides"
            )
        if r_fast + r_slow > 0:
            warnings.append(
                f"RIGHT: {r_fast} implausibly-fast, {r_slow} implausibly-slow strides"
            )

        # --- Step times + symmetry ---
        right_steps, left_steps = compute_step_times(left_times, right_times)
        if len(right_steps) < 2 or len(left_steps) < 2:
            if verbose:
                print("⚠️  Not enough step pairs!")
            return None

        n = min(len(right_steps), len(left_steps))
        right_steps = right_steps[:n]
        left_steps = left_steps[:n]

        si, per_stride = compute_symmetry_index(right_steps, left_steps,
                                                 signed=self.signed)

        # --- Spring penalty ---
        penalty = 0.0
        if self._R is not None and self._L0 is not None:
            penalty = compute_spring_penalty(
                R=self._R, L0=self._L0,
                pf_zone=self.pf_zone, df_angle=self.df_angle,
                lambda_pf=self.lambda_pf, mu_df=self.mu_df,
            )

        total_cost = si + penalty

        if verbose:
            print(f"\n   Right steps (L→R): {len(right_steps)}, "
                  f"mean = {right_steps.mean():.3f}s")
            print(f"   Left  steps (R→L): {len(left_steps)}, "
                  f"mean = {left_steps.mean():.3f}s")
            print(f"   Symmetry:          {si:+.2f}%  "
                  f"({'signed' if self.signed else 'unsigned'})")
            if self._R is not None:
                print(f"   Spring penalty:    {penalty:.4f}  "
                      f"(R={self._R:.4f}, L0={self._L0:.4f})")
            print(f"\n✅ TOTAL COST = {total_cost:.4f}\n" + "=" * 60 + "\n")

        return TrialAnalysis(
            total_cost=total_cost,
            symmetry_index=si,
            spring_penalty=penalty,
            right_step_times=right_steps,
            left_step_times=left_steps,
            per_stride_si=per_stride,
            left_heel_strikes=left_times,
            right_heel_strikes=right_times,
            left_result=left_result,
            right_result=right_result,
            warnings=warnings,
        )

    # ----- single-sensor fallback (sternum) ------------------------------

    def _analyze_single_sensor(self, xdf_path: str,
                               verbose: bool = True) -> Optional[TrialAnalysis]:
        """Sternum-based fallback. Less accurate (no per-foot labeling).

        Kept for backward compatibility with old data collection configs.
        New experiments should use two shank sensors.
        """
        stream = load_polar_stream(xdf_path, 'polar accel')
        if stream is None:
            if verbose:
                print("❌ No polar accel stream found")
            return None

        result = detect_heelstrikes_full(stream.accel, stream.timestamps,
                                         cfg=self.detection_cfg)
        if len(result.heel_strike_times) < 4:
            return None

        trial_start = stream.timestamps[0]
        trial_end = stream.timestamps[-1]
        hs = trim_peaks(result.heel_strike_times, trial_start, trial_end, self.trim_seconds)
        if len(hs) < 4:
            return None

        intervals = np.diff(hs)
        right_steps = intervals[0::2]
        left_steps = intervals[1::2]
        n = min(len(right_steps), len(left_steps))
        if n < 2:
            return None
        right_steps = right_steps[:n]
        left_steps = left_steps[:n]

        si, per_stride = compute_symmetry_index(right_steps, left_steps,
                                                 signed=self.signed)

        penalty = 0.0
        if self._R is not None and self._L0 is not None:
            penalty = compute_spring_penalty(
                R=self._R, L0=self._L0,
                pf_zone=self.pf_zone, df_angle=self.df_angle,
                lambda_pf=self.lambda_pf, mu_df=self.mu_df,
            )

        return TrialAnalysis(
            total_cost=si + penalty,
            symmetry_index=si,
            spring_penalty=penalty,
            right_step_times=right_steps,
            left_step_times=left_steps,
            per_stride_si=per_stride,
            left_heel_strikes=np.array([]),
            right_heel_strikes=hs,
            left_result=None,
            right_result=result,
            warnings=["Single-sensor mode: step alternation assumed, "
                     "per-foot labeling unreliable"],
        )

    # ----- HIL_Exo compatibility shim ------------------------------------

    def extract_data(self, trial_num: int = 1
                     ) -> Tuple[Optional[list], Optional[float]]:
        """Shim kept for backward compat with HIL_Exo's expected interface."""
        cost = self.extract_cost_from_file(trial_num)
        if cost is None:
            return None, None
        return [cost], 1.0


__all__ = [
    "SymmetryCost",
    "TrialAnalysis",
    "compute_exo_torque",
    "compute_torque_curve",
    "compute_spring_penalty",
]
