"""
hitlo.index_table — the search space.

BO optimizes ONE number: x in [-1, +1].
  x = -1  max dorsiflexor stiffness (limited by the 24-band ceiling)
  x =  0  zero torque
  x = +1  max plantarflexor stiffness (limited by the 80 Nm cap)

Each x corresponds to a row of index_unified.csv holding four physical
parameters (R, theta, L0, attachment_ratio) that the operator sets on the
device. The table is built and validated by build_index_unified.m.

TWO RULES, both load-bearing:

1. NEVER INTERPOLATE between rows. Adjacent rows are independent survivors
   of a 4-D sweep and can differ wildly in every parameter. Averaging two of
   them yields a configuration that never passed the sign, cap, engagement,
   or edge-cliff filters, and may exceed the torque cap. Always snap to a row.

2. x IS NOT UNIFORMLY SPACED. DF steps are 1/nPerArmDF, PF steps are
   1/nPerArmPF, and those differ. Match on nearest value in the x column;
   never compute a row position arithmetically from x.

Because every row is pre-validated, there is no runtime safety check here.
Safety lives in the builder. If you change the builder's caps or filters,
regenerate the CSV — do not add checks on this side.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "x", "direction", "R", "theta", "L0", "attachment_ratio",
    "stiff_Nm_per_deg", "stiff_Nm_per_rad", "frac_of_ceiling",
    "dose_signed_Nm", "peak_full_grid_Nm", "engage_deg",
]


class IndexTable:
    """The 1-D search space, loaded from index_unified.csv."""

    def __init__(self, csv_path: str) -> None:
        path = Path(csv_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Index table not found: {path}\n"
                f"Run build_index_unified.m and copy index_unified.csv here."
            )
        self.path = path
        self.df = pd.read_csv(path)

        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {missing}\n"
                f"This is probably an old two-table CSV (index_PF/index_DF) "
                f"or a stale build. Regenerate with build_index_unified.m."
            )

        self.df = self.df.sort_values("x").reset_index(drop=True)
        self._x = self.df["x"].to_numpy(dtype=float)

        self._validate()

    # ------------------------------------------------------------------
    # Validation — catches a stale or hand-edited CSV at load, not mid-trial
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if len(self.df) < 3:
            raise ValueError(f"Index table has only {len(self.df)} rows.")

        if not np.all(np.diff(self._x) > 0):
            raise ValueError("x column is not strictly increasing after sort — "
                             "duplicate x values in the table.")

        if not np.isclose(self._x[0], -1.0) or not np.isclose(self._x[-1], 1.0):
            raise ValueError(f"x should span [-1, +1]; got "
                             f"[{self._x[0]:.4f}, {self._x[-1]:.4f}].")

        zero_rows = self.df[self.df["direction"] == 0]
        if len(zero_rows) != 1:
            raise ValueError(f"Expected exactly one direction==0 row, "
                             f"found {len(zero_rows)}.")

        # Stiffness must be monotone across the whole axis — it is the sort key.
        st = self.df["stiff_Nm_per_deg"].to_numpy(dtype=float)
        if np.any(np.diff(st) < -1e-9):
            raise ValueError("stiff_Nm_per_deg is not monotone across x. "
                             "The table is not correctly ordered.")

        # Every non-zero row needs all four physical parameters.
        phys = self.df.loc[self.df["direction"] != 0,
                           ["R", "theta", "L0", "attachment_ratio"]]
        if phys.isna().any().any():
            raise ValueError("NaN in physical parameters on a non-zero row.")

    # ------------------------------------------------------------------
    # The one operation BO needs
    # ------------------------------------------------------------------

    def snap(self, x: float) -> float:
        """Return the nearest x value that actually exists in the table.

        Call this BEFORE running the trial, and record the snapped value as
        the trial's location. Recording the raw BO proposal would tell the GP
        it sampled a point that was never set on the device.
        """
        return float(self._x[int(np.argmin(np.abs(self._x - float(x))))])

    def row(self, x: float) -> Dict:
        """Snap x to a row and return it as a dict.

        For the zero row the four physical parameters come back as None:
        x = 0 is NOT "device off". The bands are always on, so the controller
        must actively cancel them (tau_cable = 0 - tau_band).
        """
        idx = int(np.argmin(np.abs(self._x - float(x))))
        r = self.df.iloc[idx]
        is_zero = int(r["direction"]) == 0

        return {
            "x": float(r["x"]),
            "direction": int(r["direction"]),
            "R": None if is_zero else float(r["R"]),
            "theta": None if is_zero else float(r["theta"]),
            "L0": None if is_zero else float(r["L0"]),
            "attach": None if is_zero else float(r["attachment_ratio"]),
            "stiff_Nm_per_rad": float(r["stiff_Nm_per_rad"]),
            "dose_Nm": float(r["dose_signed_Nm"]),
            "peak_full_Nm": float(r["peak_full_grid_Nm"]),
            "engage_deg": None if is_zero else float(r["engage_deg"]),
            "is_zero": is_zero,
        }

    @property
    def x_values(self) -> np.ndarray:
        """All reachable x values.

        Evaluate the acquisition function on THESE points and take the argmax,
        rather than optimizing continuously and snapping afterward. Then BO
        cannot propose a configuration that does not exist.
        """
        return self._x.copy()

    def describe(self, x: float) -> str:
        """One-line human summary, for logs."""
        r = self.row(x)
        if r["is_zero"]:
            return f"x={r['x']:+.4f}  ZERO (controller must cancel bands)"
        d = "PF" if r["direction"] > 0 else "DF"
        return (f"x={r['x']:+.4f}  {d}  "
                f"R={r['R']:.4f} m  theta={r['theta']:.2f} deg  "
                f"L0={r['L0']:.4f} m  attach={r['attach']:+.4f}  "
                f"[{r['stiff_Nm_per_rad']:+.1f} Nm/rad, "
                f"dose {r['dose_Nm']:+.2f} Nm]")

    def __len__(self) -> int:
        return len(self.df)


__all__ = ["IndexTable"]