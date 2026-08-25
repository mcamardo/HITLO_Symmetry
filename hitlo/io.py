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

# BIDS modality per backend. Polar recordings were filed under 'eeg' (an
# artefact of the LabRecorder template used at the time, not a claim about
# the data); Trigno IMU recordings are filed under 'motion', which is what
# BIDS actually specifies for them. Both must keep working: existing sessions
# are full of eeg-suffixed files.
_MODALITY = {'polar': 'eeg', 'trigno': 'motion'}


def backend_modality(config: Optional[dict] = None) -> str:
    """'eeg' or 'motion' — the BIDS modality this backend records under."""
    return _MODALITY.get(sensing_config(config).get('backend'), 'eeg')


def trial_filename(subject: str, session: str, run: int,
                   task: str = "Default",
                   modality: str = "eeg") -> str:
    """BIDS-style XDF filename for a trial.

    trial_filename("P048", "S001", 7)
        -> "sub-P048_ses-S001_task-Default_run-007_eeg.xdf"
    trial_filename("P012", "S001", 7, modality="motion")
        -> "sub-P012_ses-S001_task-Default_run-007_motion.xdf"

    `modality` defaults to 'eeg' so every existing caller is unaffected.
    """
    return (f"sub-{subject}_ses-{session}_task-{task}_run-{run:03d}"
            f"_{modality}.xdf")


def trial_dir(config: dict) -> "Path":
    """Directory this backend's recordings live in, per BIDS modality."""
    from pathlib import Path
    subj = config['Subject']
    return (Path(subj['base_dir']) / f"sub-{subj['id']}" /
            f"ses-{subj['session']}" / backend_modality(config))


def trial_path(config: dict, run: int, task: str = "Default") -> "Path":
    """Full path to a trial recording, correct for the configured backend."""
    return trial_dir(config) / trial_filename(
        config['Subject']['id'], config['Subject']['session'], run, task,
        modality=backend_modality(config))


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

# Body segments a sensor can be on. 'shank' is the default when a label names
# no segment, so the original two-sensor labels (left_acc_x) keep working.
_SEGMENTS = ('shank', 'foot', 'thigh')
_DEFAULT_SEGMENT = 'shank'


def _label_segment(label: str) -> str:
    """Which body segment a channel label refers to."""
    for seg in _SEGMENTS:
        if seg in label:
            return seg
    return _DEFAULT_SEGMENT


def _match_axis(label: str, side: str, segment: str,
                kind_tokens: Sequence[str]) -> Optional[str]:
    """Axis ('x'/'y'/'z') if `label` is this side+segment+kind, else None."""
    if not label.startswith(side):
        return None
    if _label_segment(label) != segment:
        return None
    if not any(tok in label for tok in kind_tokens):
        return None
    for ax in _AXES:
        if label.endswith(ax):
            return ax
    return None


def _demux(labels: List[str], series: np.ndarray, side: str,
           segment: str = _DEFAULT_SEGMENT
           ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Pull (accel, gyro) as (N,3) for one side+segment out of the stream.

    Raises on an ambiguous label set rather than picking whichever column
    comes first. Adding a foot sensor labelled 'right_foot_acc_x' alongside a
    shank sensor labelled 'right_acc_x' would otherwise blend the two
    silently, in channel order -- the kind of failure that produces entirely
    plausible numbers from the wrong body segment.
    """
    def gather(tokens: Sequence[str]) -> Optional[np.ndarray]:
        cols: Dict[str, List[int]] = {}
        for i, lab in enumerate(labels):
            ax = _match_axis(lab, side, segment, tokens)
            if ax is not None:
                cols.setdefault(ax, []).append(i)
        dupes = {ax: idxs for ax, idxs in cols.items() if len(idxs) > 1}
        if dupes:
            detail = "; ".join(
                f"{ax}: " + ", ".join(labels[i] for i in idxs)
                for ax, idxs in dupes.items())
            raise ValueError(
                f"ambiguous channel labels for {side} {segment} -- {detail}. "
                f"Give each sensor a distinct segment in its label "
                f"(e.g. right_shank_acc_x vs right_foot_acc_x) so they cannot "
                f"be confused.")
        if len(cols) != 3:
            return None
        return series[:, [cols['x'][0], cols['y'][0], cols['z'][0]]]

    return gather(_ACC_TOKENS), gather(_GYR_TOKENS)


def load_trigno_segment(xdf_path: str,
                        side: str,
                        segment: str = _DEFAULT_SEGMENT,
                        stream_name: str = "TrignoIMU",
                        ) -> Optional[SensorStream]:
    """One side+segment out of the multiplexed stream.

    Use for sensors beyond the two shanks -- a foot sensor, for instance,
    which with its shank counterpart gives ankle angle.
    """
    data = _load_xdf(xdf_path)
    if data is None:
        return None
    for stream in data:
        if stream['info']['name'][0] != stream_name:
            continue
        series = np.asarray(stream['time_series'], dtype=np.float64)
        timestamps = np.asarray(stream['time_stamps'])
        if len(timestamps) < 2 or series.ndim != 2:
            return None
        labels = _channel_labels(stream)
        if not labels or len(labels) != series.shape[1]:
            return None
        acc, gyr = _demux(labels, series, side, segment)
        if acc is None:
            return None
        return SensorStream(
            accel=acc, timestamps=timestamps,
            actual_fs=_measured_fs(timestamps),
            name=f"{stream_name}:{side}_{segment}",
            gyro=gyr, side=side, backend="trigno")
    return None


def trigno_inventory(xdf_path: str,
                     stream_name: str = "TrignoIMU") -> Dict[str, List[str]]:
    """What sensors a Trigno recording actually contains.

    Returns {'left': ['shank'], 'right': ['shank', 'foot'], ...}. Useful for
    checking a bridge change did what was intended before recording a session
    on it.
    """
    data = _load_xdf(xdf_path)
    out: Dict[str, List[str]] = {}
    if data is None:
        return out
    for stream in data:
        if stream['info']['name'][0] != stream_name:
            continue
        for lab in _channel_labels(stream):
            for side in _DEFAULT_SIDES:
                if lab.startswith(side):
                    seg = _label_segment(lab)
                    out.setdefault(side, [])
                    if seg not in out[side]:
                        out[side].append(seg)
    return out


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
    "load_trigno_segment",
    "trigno_inventory",
    "backend_modality",
    "trial_dir",
    "trial_path",
    "PolarStream",
    "trial_filename",
    "load_polar_stream",
    "load_both_polar_streams",
    "load_trigno_streams",
    "load_streams",
    "live_stream_names",
    "sensing_config",
]
