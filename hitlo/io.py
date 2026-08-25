"""
hitlo.io — data loading and file-path conventions.

Centralizes XDF loading and trial-file naming so that the rest of the code
doesn't have to care about LSL stream names or BIDS-style paths.

TWO SENSOR BACKENDS
-------------------
polar   Two Polar H10 straps, each publishing its own LSL stream
        ('polar accel left' / 'polar accel right'), accelerometer only,
        nominally 200 Hz.

trigno  Delsys Trigno Avanti IMUs bridged to a SINGLE LSL stream (default
        name 'TrignoIMU') carrying every sensor's accel AND gyro, with the
        side encoded in the channel labels (left_gyr_x, right_acc_z, ...),
        nominally 148 Hz.

The structural difference is the reason this module grew a dispatch layer:
Polar gives two streams to resolve by name, Trigno gives one stream to
demultiplex by channel label. Both normalize into the same SensorStream so
nothing downstream has to know which produced it.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
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
# The normalized per-limb stream
# ===========================================================================

@dataclass
class SensorStream:
    """One limb's data, whatever hardware produced it.

    `accel` is always present. `gyro` is None for accelerometer-only
    backends (Polar), so a consumer that needs it must check — the gyro
    detector raises a clear error rather than silently degrading.
    """
    accel: np.ndarray                  # shape (N, 3)
    timestamps: np.ndarray             # shape (N,), LSL seconds
    actual_fs: float                   # measured from the timestamps
    name: str                          # source stream name
    gyro: Optional[np.ndarray] = None  # shape (N, 3) or None
    side: Optional[str] = None         # 'left' | 'right' | None
    backend: str = "polar"             # which loader produced this

    @property
    def has_gyro(self) -> bool:
        return self.gyro is not None and len(self.gyro) == len(self.accel)


# Back-compat: this name is used across the Polar-era code and by saved
# analysis scripts outside the repo. Keep it working.
PolarStream = SensorStream


# ===========================================================================
# Shared XDF helpers
# ===========================================================================

def _load_xdf(xdf_path: str) -> Optional[list]:
    if not os.path.exists(xdf_path):
        return None
    try:
        import pyxdf
        data, _ = pyxdf.load_xdf(xdf_path)
        return data
    except Exception:
        return None


def _channel_labels(stream) -> List[str]:
    """Channel labels for an XDF stream, lowercased. [] if not declared."""
    try:
        chans = stream['info']['desc'][0]['channels'][0]['channel']
        return [c['label'][0].strip().lower() for c in chans]
    except Exception:
        return []


def _measured_fs(timestamps: np.ndarray) -> float:
    return 1.0 / float(np.median(np.diff(timestamps)))


# ===========================================================================
# Polar backend — two streams, accelerometer only
# ===========================================================================

def load_polar_stream(xdf_path: str,
                      stream_name: str
                      ) -> Optional[SensorStream]:
    """Load a single Polar accelerometer stream from an XDF file.

    Returns None if the file is missing or the stream is absent, so callers
    can check once and fall back (e.g. single-sensor mode).
    """
    data = _load_xdf(xdf_path)
    if data is None:
        return None

    for stream in data:
        if stream['info']['name'][0] == stream_name:
            # float64 regardless of the stream's channel_format —
            # int16 accel overflows as soon as anything squares it.
            accel = np.asarray(stream['time_series'], dtype=np.float64)
            timestamps = np.asarray(stream['time_stamps'])
            if len(timestamps) < 2:
                return None
            side = ('left' if stream_name.endswith('left')
                    else 'right' if stream_name.endswith('right') else None)
            return SensorStream(
                accel=accel,
                timestamps=timestamps,
                actual_fs=_measured_fs(timestamps),
                name=stream_name,
                gyro=None,
                side=side,
                backend="polar",
            )

    return None


def load_both_polar_streams(xdf_path: str
                            ) -> Tuple[Optional[SensorStream], Optional[SensorStream]]:
    """Convenience: load the left + right shank streams in one call."""
    left = load_polar_stream(xdf_path, 'polar accel left')
    right = load_polar_stream(xdf_path, 'polar accel right')
    return left, right


# ===========================================================================
# Trigno backend — ONE stream, demultiplexed by channel label
# ===========================================================================

_DEFAULT_SIDES = ('left', 'right')
_ACC_TOKENS = ('acc', 'accel')
_GYR_TOKENS = ('gyr', 'gyro')
_AXES = ('x', 'y', 'z')


def _match_axis(label: str, side: str,
                kind_tokens: Sequence[str]) -> Optional[str]:
    """Return the axis ('x'/'y'/'z') if `label` is this side+kind, else None."""
    if not label.startswith(side):
        return None
    if not any(tok in label for tok in kind_tokens):
        return None
    for ax in _AXES:
        if label.endswith(ax):
            return ax
    return None


def _demux(labels: List[str], series: np.ndarray, side: str
           ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Pull (accel, gyro) as (N,3) for one side out of a multiplexed stream."""
    def gather(tokens: Sequence[str]) -> Optional[np.ndarray]:
        cols: Dict[str, int] = {}
        for i, lab in enumerate(labels):
            ax = _match_axis(lab, side, tokens)
            if ax is not None and ax not in cols:
                cols[ax] = i
        if len(cols) != 3:
            return None
        return series[:, [cols['x'], cols['y'], cols['z']]]

    return gather(_ACC_TOKENS), gather(_GYR_TOKENS)


def load_trigno_streams(xdf_path: str,
                        stream_name: str = "TrignoIMU",
                        sides: Sequence[str] = _DEFAULT_SIDES,
                        ) -> Tuple[Optional[SensorStream], Optional[SensorStream]]:
    """Load left/right SensorStreams from a single multiplexed Trigno stream.

    Unlike the Polar backend there is only ONE LSL stream in the file; the
    limb is encoded in the channel labels, so the side split happens here
    rather than at stream-resolution time.

    Refuses to guess: if the stream declares no channel labels, or the label
    count does not match the data width, this returns (None, None) rather
    than assuming a column order. A wrong column order would silently swap
    left and right, which inverts the sign of the symmetry index.

    Returns (left, right); either may be None if that side's channels are
    absent or incomplete.
    """
    data = _load_xdf(xdf_path)
    if data is None:
        return None, None

    for stream in data:
        if stream['info']['name'][0] != stream_name:
            continue

        series = np.asarray(stream['time_series'], dtype=np.float64)
        timestamps = np.asarray(stream['time_stamps'])
        if len(timestamps) < 2 or series.ndim != 2:
            return None, None
        labels = _channel_labels(stream)
        if not labels or len(labels) != series.shape[1]:
            return None, None

        fs = _measured_fs(timestamps)
        out: List[Optional[SensorStream]] = []
        for side in sides:
            acc, gyr = _demux(labels, series, side)
            if acc is None:
                out.append(None)
                continue
            out.append(SensorStream(
                accel=acc,
                timestamps=timestamps,
                actual_fs=fs,
                name=f"{stream_name}:{side}",
                gyro=gyr,
                side=side,
                backend="trigno",
            ))
        return (out[0], out[1]) if len(out) == 2 else (None, None)

    return None, None


# ===========================================================================
# Backend dispatch
# ===========================================================================

def sensing_config(config: Optional[dict]) -> dict:
    """Read the Sensing block, defaulting to the Polar behaviour.

    Permissive on purpose: every config written before this block existed —
    which is every saved session config — must keep working unchanged.
    """
    s = dict((config or {}).get('Sensing') or {})
    s.setdefault('backend', 'polar')
    s.setdefault('stream',
                 'TrignoIMU' if s['backend'] == 'trigno' else 'polar accel')
    return s


def load_streams(xdf_path: str, config: Optional[dict] = None
                 ) -> Tuple[Optional[SensorStream], Optional[SensorStream]]:
    """Load (left, right) for whichever backend the config names.

    This is the entry every consumer should use. `load_both_polar_streams`
    remains for the Polar path and for older scripts.
    """
    s = sensing_config(config)
    if s['backend'] == 'trigno':
        return load_trigno_streams(xdf_path, stream_name=s['stream'])
    return load_both_polar_streams(xdf_path)


def live_stream_names(config: Optional[dict] = None) -> List[str]:
    """LSL stream names to resolve for a live session, per backend.

    Polar publishes one stream per side; Trigno publishes a single stream
    covering both. Consumers that show per-side status need to branch on the
    length of this list.
    """
    s = sensing_config(config)
    if s['backend'] == 'trigno':
        return [s['stream']]
    return ['polar accel left', 'polar accel right']


__all__ = [
    "SensorStream",
    "PolarStream",
    "trial_filename",
    "load_polar_stream",
    "load_both_polar_streams",
    "load_trigno_streams",
    "load_streams",
    "live_stream_names",
    "sensing_config",
]
