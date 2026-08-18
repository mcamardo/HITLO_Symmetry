"""
hitlo.cost — BO cost function: step-time symmetry.

Version 3.0.0 — torque model removed.

WHAT CHANGED AND WHY
This module used to carry its own forward model of the spring (compute_exo_torque,
compute_torque_curve) plus a shape penalty (compute_spring_penalty). All three are
gone.

The torque model was a second, drifted copy of the physics: it hardcoded spring
rate 10500 N/m, R = 0.28 m, and theta = 196 deg, while the index table is built
at 12000 N/m with R swept over [0.15, 0.45] and theta over [110, 215]. Almost no
row of the table can be represented by that function, so any torque it returned
described a device that does not exist. The single source of truth for the
physics is now build_index_unified.m and the CSV it writes; if you need a torque
curve for a configuration, read it from the table or regenerate it in MATLAB.

The shape penalty is gone because the index table makes it redundant. It existed
to push BO away from configurations with the wrong torque shape — wrong sign in a
zone, engaging at the wrong angle. Every row of the table already passed those
filters at build time, so there is nothing left to penalize. It was also inert in
practice: lambda_pf and mu_df have defaulted to 0.0 since the signed-SI paradigm
landed, so total_cost has equaled the symmetry index for some time.

The cost is now exactly the step-time symmetry index. Raw and signed; the
target-distance transformation |SI - si_target| is applied downstream in
hitlo.hil_exo._mean_normalize_y, so per-trial logs show real SI values rather
than distances.
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
# Per-trial analysis result (returned for inspection / QC / logging)
# ===========================================================================

@dataclass
class TrialAnalysis:
    """Everything computed for one trial. BO uses `symmetry_index`; QC uses the rest.

    `total_cost` is the raw (signed, if signed=True) symmetry index. The
    target-distance transformation |SI - si_target| happens DOWNSTREAM in
    hitlo.hil_exo._mean_normalize_y, not here, so per-trial logging shows
    actual SI values rather than distances-from-target.

    `spring_penalty` is retained at 0.0 for backward compatibility with older
    result files and any downstream code that still reads the field. Nothing
    writes a nonzero value anymore.
    """
    # Headline numbers
    total_cost: float           # == symmetry_index (raw, signed)
    symmetry_index: float       # signed if signed=True, else |SI|
    spring_penalty: float = 0.0  # deprecated, always 0.0

    # Step times
    right_step_times: np.ndarray = field(default_factory=lambda: np.array([]))
    left_step_times: np.ndarray = field(default_factory=lambda: np.array([]))
    per_stride_si: np.ndarray = field(default_factory=lambda: np.array([]))

    # Heel strike times (post-trim, post-plausibility)
    left_heel_strikes: np.ndarray = field(default_factory=lambda: np.array([]))
    right_heel_strikes: np.ndarray = field(default_factory=lambda: np.array([]))

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

    Reports raw signed SI per trial. The target-distance transformation
    (|SI - si_target|) is applied downstream in hitlo.hil_exo. The si_target
    field here is stored only for reference (logging / introspection); it does
    NOT modify the returned cost value.

    This class knows nothing about the exoskeleton configuration. Which
    configuration produced a trial is the optimizer's business, tracked as an
    index value in hitlo.hil_exo against hitlo.index_unified.IndexTable.

    Usage
    -----
    >>> cost = SymmetryCost(trial_data_dir="/path/to/xdf/files",
    ...                     subject_id="P062", session="S001",
    ...                     signed=True, si_target=-3.0, trim_seconds=3.0)
    >>> total_cost = cost.extract_cost_from_file(trial_num=1)
    """

    def __init__(self,
                 trial_data_dir: str,
                 subject_id: str = "",
                 session: str = "",
                 detection_cfg: Optional[DetectionConfig] = None,
                 # symmetry behavior
                 signed: bool = False,
                 si_target: float = 0.0,
                 trim_seconds: float = 3.0,
                 ):
        self.cost_name = "gait_symmetry"
        self.trial_data_dir = trial_data_dir
        self.subject_id = subject_id
        self.session = session

        self.detection_cfg = detection_cfg or DetectionConfig()

        self.signed = signed
        self.si_target = si_target
        self.trim_seconds = trim_seconds

        # Why the last analyze_trial returned None. Every failure path sets
        # this, so a caller can tell "sensor stream missing" apart from
        # "subject barely walked" instead of printing one generic message.
        self.last_failure: Optional[str] = None
        self.last_warnings: List[str] = []

        print("SymmetryCost v3.0.0 initialized (symmetry index only)")
        print(f"   Mode:              {'SIGNED' if signed else 'UNSIGNED (abs)'}")
        if signed:
            print(f"   SI target:         {si_target:+.1f}%  "
                  f"(BO drives SI toward this; applied in hil_exo)")
        print(f"   Directory:         {trial_data_dir}")
        print(f"   Detection:         cluster-keep-last + stance confirm")
        print(f"   Jerk threshold:    {self.detection_cfg.strict_thresh:.2f} SD strict / "
              f"{self.detection_cfg.recovery_thresh:.2f} SD recovery")
        print(f"   Cluster gap:       {self.detection_cfg.cluster_gap_s}s")
        print(f"   Stance check:      {self.detection_cfg.stance_duration_s}s window, "
              f"±{self.detection_cfg.stance_tolerance_pct * 100:.0f}% of baseline")
        if trim_seconds > 0:
            print(f"   Steady-state trim: {trim_seconds:.1f}s from each end")

    def _fail(self, reason: str, verbose: bool) -> None:
        """Record why analysis failed, print it, and return None."""
        self.last_failure = reason
        if verbose:
            print(f"ANALYSIS FAILED: {reason}")
        return None

    # ----- main entry points ---------------------------------------------

    def extract_cost_from_file(self,
                               trial_num: int,
                               filename: Optional[str] = None
                               ) -> Optional[float]:
        """Analyze one trial and return the raw (signed) symmetry index.

        The downstream BO loop in hitlo.hil_exo converts this to a
        distance-from-target via |cost - si_target| before feeding the GP.

        For richer access (heel strikes, per-stride SI, etc.) use
        `analyze_trial()` directly.
        """
        analysis = self.analyze_trial(trial_num=trial_num, filename=filename)
        if analysis is None:
            return None
        # Non-fatal problems (implausible strides, single-sensor fallback) are
        # attached to the analysis and were previously dropped on the floor here,
        # so the console never saw them. Keep them reachable.
        self.last_warnings = list(analysis.warnings)
        return analysis.total_cost

    def analyze_trial(self,
                      trial_num: int,
                      filename: Optional[str] = None,
                      verbose: bool = True,
                      ) -> Optional[TrialAnalysis]:
        """Full analysis of one trial; returns everything needed for QC + BO."""
        self.last_failure = None
        if filename is None:
            filename = trial_filename(self.subject_id, self.session, trial_num)
        xdf_path = os.path.join(self.trial_data_dir, filename)

        if verbose:
            print("\n" + "=" * 60)
            print(f"ANALYZING TRIAL {trial_num}")
            print("=" * 60)
            print(f"Loading {xdf_path}...")

        left, right = load_both_polar_streams(xdf_path)

        if left is None or right is None:
            if verbose:
                print("Two-sensor streams not found — falling back to single sensor")
            return self._analyze_single_sensor(xdf_path, verbose=verbose)

        if verbose:
            print(f"   Left:  {len(left.accel)} samples, {left.actual_fs:.2f} Hz")
            print(f"   Right: {len(right.accel)} samples, {right.actual_fs:.2f} Hz")

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
                  f"-> {len(left_result.heel_strike_indices)} heel strikes")
            print(f"   Right: {len(right_result.strict_peaks)} strict "
                  f"+ {len(right_result.recovered_peaks)} recovered "
                  f"= {len(right_result.all_candidates)} candidates "
                  f"-> {len(right_result.heel_strike_indices)} heel strikes")

        left_times_raw = left_result.heel_strike_times
        right_times_raw = right_result.heel_strike_times

        if len(left_times_raw) < 3 or len(right_times_raw) < 3:
            return self._fail(
                f"Too few heel strikes detected (left={len(left_times_raw)}, "
                f"right={len(right_times_raw)}, need >=3 each). The recording "
                f"loaded fine, so this is detection, not I/O: either the subject "
                f"barely walked, or a sensor lost skin contact mid-trial.",
                verbose)

        # --- Trim + plausibility filter ---
        trial_start = min(left.timestamps[0], right.timestamps[0])
        trial_end = max(left.timestamps[-1], right.timestamps[-1])
        left_times = trim_peaks(left_times_raw, trial_start, trial_end, self.trim_seconds)
        right_times = trim_peaks(right_times_raw, trial_start, trial_end, self.trim_seconds)

        if self.trim_seconds > 0 and verbose:
            print(f"   Trimmed ±{self.trim_seconds}s: "
                  f"Left {len(left_times_raw)}->{len(left_times)}, "
                  f"Right {len(right_times_raw)}->{len(right_times)}")

        if len(left_times) < 3 or len(right_times) < 3:
            return self._fail(
                f"Too few heel strikes survived the +/-{self.trim_seconds}s trim "
                f"(left={len(left_times)}, right={len(right_times)}). The trial "
                f"is probably shorter than the trim window allows.", verbose)

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
            return self._fail(
                f"Heel strikes found but they do not interleave into steps "
                f"(right_steps={len(right_steps)}, left_steps={len(left_steps)}). "
                f"Usually means one sensor's strikes are all on one side of the "
                f"other's in time -- check the two streams overlap.", verbose)

        n = min(len(right_steps), len(left_steps))
        right_steps = right_steps[:n]
        left_steps = left_steps[:n]

        si, per_stride = compute_symmetry_index(right_steps, left_steps,
                                                signed=self.signed)

        # The cost IS the symmetry index. No shape penalty: every configuration
        # BO can reach already satisfies the shape constraints by construction,
        # because it is a row of the pre-filtered index table.
        total_cost = si

        if verbose:
            print(f"\n   Right steps (L->R): {len(right_steps)}, "
                  f"mean = {right_steps.mean():.3f}s")
            print(f"   Left  steps (R->L): {len(left_steps)}, "
                  f"mean = {left_steps.mean():.3f}s")
            print(f"   Symmetry:           {si:+.2f}%  "
                  f"({'signed' if self.signed else 'unsigned'})")
            if self.signed:
                print(f"   |SI - target|:      {abs(si - self.si_target):.2f}%  "
                      f"(target = {self.si_target:+.1f}%)")
            print(f"\nTOTAL COST = {total_cost:.4f}\n" + "=" * 60 + "\n")

        return TrialAnalysis(
            total_cost=total_cost,
            symmetry_index=si,
            spring_penalty=0.0,
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
            return self._fail(
                "No usable accel stream in the file. Expected 'polar accel left' "
                "and 'polar accel right'; neither those nor a single 'polar accel' "
                "were present. Check what LabRecorder actually had checked.",
                verbose)

        result = detect_heelstrikes_full(stream.accel, stream.timestamps,
                                         cfg=self.detection_cfg)
        if len(result.heel_strike_times) < 4:
            return self._fail(
                f"Single-sensor mode: only {len(result.heel_strike_times)} heel "
                f"strikes detected (need >=4).", verbose)

        hs = trim_peaks(result.heel_strike_times,
                        stream.timestamps[0], stream.timestamps[-1],
                        self.trim_seconds)
        if len(hs) < 4:
            return self._fail(
                f"Single-sensor mode: only {len(hs)} heel strikes survived trim.",
                verbose)

        intervals = np.diff(hs)
        right_steps = intervals[0::2]
        left_steps = intervals[1::2]
        n = min(len(right_steps), len(left_steps))
        if n < 2:
            return self._fail(
                "Single-sensor mode: fewer than 2 complete stride pairs.", verbose)
        right_steps = right_steps[:n]
        left_steps = left_steps[:n]

        si, per_stride = compute_symmetry_index(right_steps, left_steps,
                                                signed=self.signed)

        warnings_list = ["Single-sensor mode: step alternation assumed, "
                         "per-foot labeling unreliable"]
        if self.signed and self.si_target != 0.0:
            warnings_list.append(
                "Single-sensor mode with nonzero si_target: signed SI direction "
                "unreliable; results should be interpreted cautiously."
            )

        return TrialAnalysis(
            total_cost=si,
            symmetry_index=si,
            spring_penalty=0.0,
            right_step_times=right_steps,
            left_step_times=left_steps,
            per_stride_si=per_stride,
            left_heel_strikes=np.array([]),
            right_heel_strikes=hs,
            left_result=None,
            right_result=result,
            warnings=warnings_list,
        )

    # ----- HIL_Exo compatibility shim ------------------------------------

    def extract_data(self, trial_num: int = 1
                     ) -> Tuple[Optional[list], Optional[float]]:
        """Shim kept for backward compat with HIL_Exo's expected interface."""
        cost = self.extract_cost_from_file(trial_num)
        if cost is None:
            return None, None
        return [cost], 1.0


__all__ = ["SymmetryCost", "TrialAnalysis"]