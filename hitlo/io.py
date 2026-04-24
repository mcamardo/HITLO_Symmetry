"""
hitlo.io — data loading and file-path conventions.

Centralizes XDF loading and trial-file naming so that the rest of the code
doesn't have to care about LSL stream names or BIDS-style paths.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import os
import numpy as np


# ===========================================================================
# Trial file naming (BIDS-ish convention used throughout the project)
# ===========================================================================

def trial_filename(subject: str, session: str, run: int,
                   task: str = "Default") -> str:
    """BIDS-style XDF filename for a trial.

    Example: trial_filename("P048", "S001", 7)
             -> "sub-P048_ses-S001_task-Default_run-007_eeg.xdf"
    """
    return f"sub-{subject}_ses-{session}_task-{task}_run-{run:03d}_eeg.xdf"


# ===========================================================================
# XDF stream loading
# ===========================================================================

@dataclass
class PolarStream:
    """One sensor's data loaded from XDF."""
    accel: np.ndarray       # shape (N, 3)
    timestamps: np.ndarray  # shape (N,), LSL seconds
    actual_fs: float        # measured sample rate
    name: str               # 'polar accel left' etc.


def load_polar_stream(xdf_path: str,
                      stream_name: str
                      ) -> Optional[PolarStream]:
    """Load a single Polar accelerometer stream from an XDF file.

    Returns None if the file is missing or the stream is absent, so callers
    can check once and fall back (e.g. single-sensor mode).
    """
    if not os.path.exists(xdf_path):
        return None

    try:
        import pyxdf
        data, _ = pyxdf.load_xdf(xdf_path)
    except Exception:
        return None

    for stream in data:
        if stream['info']['name'][0] == stream_name:
            accel = np.asarray(stream['time_series'])
            timestamps = np.asarray(stream['time_stamps'])
            if len(timestamps) < 2:
                return None
            actual_fs = 1.0 / float(np.median(np.diff(timestamps)))
            return PolarStream(
                accel=accel,
                timestamps=timestamps,
                actual_fs=actual_fs,
                name=stream_name,
            )

    return None


def load_both_polar_streams(xdf_path: str
                             ) -> Tuple[Optional[PolarStream], Optional[PolarStream]]:
    """Convenience: load the left + right shank streams in one call."""
    left = load_polar_stream(xdf_path, 'polar accel left')
    right = load_polar_stream(xdf_path, 'polar accel right')
    return left, right


__all__ = [
    "PolarStream",
    "trial_filename",
    "load_polar_stream",
    "load_both_polar_streams",
]
