"""
hitlo.detectors — one entry point, two detection methods.

The optimizer, the cost function and the diagnostics all want the same
thing: heel-strike times for a limb. How those were obtained — an impact
transient in accelerometer jerk, or a zero crossing in shank angular
velocity — is a property of the hardware in use, not of the experiment.

This module is the seam. Everything above it calls detect(stream, config)
and receives a DetectionResult. Everything below it is method-specific.

    accel   hitlo.detection        Polar H10, jerk peaks + clustering
    gyro    hitlo.detection_gyro   Trigno Avanti, swing peak -> zero crossing

Selection comes from the config so both paths stay live and comparable:

    Sensing:
      backend:  trigno      # polar | trigno   (which loader)
      detector: gyro        # accel | gyro     (which method)

`detector` defaults to whatever suits the backend, so a config that predates
this module keeps the accelerometer behaviour it had.

NOT INTERCHANGEABLE MEASUREMENTS
--------------------------------
The two methods find different instants. The gyro zero crossing is initial
contact; the jerk peak is the impact shock a few tens of milliseconds later.
Both are internally consistent, but a symmetry index from one cannot be
compared against a baseline collected with the other. Switching detector
mid-study means re-baselining.
"""

from typing import Optional
import numpy as np

from hitlo.detection import DetectionConfig, DetectionResult, detect_heelstrikes_full
from hitlo.detection_gyro import GyroDetectionConfig, detect_heelstrikes_gyro
from hitlo.io import SensorStream, sensing_config


def detector_name(config: Optional[dict]) -> str:
    """Which detection method the config selects: 'accel' or 'gyro'."""
    s = sensing_config(config)
    explicit = s.get('detector')
    if explicit in ('accel', 'gyro'):
        return explicit
    # Default per backend. Polar has no gyro to use; Trigno is the reason
    # the gyro path exists.
    return 'gyro' if s.get('backend') == 'trigno' else 'accel'


def detect(stream: SensorStream,
           config: Optional[dict] = None,
           cfg: Optional[object] = None,
           method: Optional[str] = None,
           ) -> DetectionResult:
    """Heel strikes for one limb, by whichever method applies.

    Parameters
    ----------
    stream : SensorStream
        One limb, from hitlo.io. Must carry gyro for the gyro method.
    config : dict, optional
        Full experiment config; the Sensing block selects the method.
    cfg : DetectionConfig | GyroDetectionConfig, optional
        Explicit detector settings. The sample rate is overridden from the
        stream regardless — see for_stream() in either config class.
    method : str, optional
        Force 'accel' or 'gyro', ignoring the config. Used by the
        comparison tooling to run both over one recording.

    Raises
    ------
    ValueError
        If the gyro method is selected for a stream that has no gyro. This
        is deliberately loud: silently falling back to the accelerometer
        would change which instant is being measured, and the resulting
        symmetry index would be quietly incomparable with the rest of the
        session.
    """
    how = method or detector_name(config)

    if how == 'gyro':
        if not getattr(stream, 'has_gyro', False):
            raise ValueError(
                f"gyro detection requested for stream '{stream.name}', which "
                f"carries no gyroscope data. Either the backend is 'polar' "
                f"(accelerometer only) or the Trigno channel labels did not "
                f"demultiplex. Refusing to fall back to the accelerometer: "
                f"that detects a different instant and would silently produce "
                f"a symmetry index incomparable with the rest of the session.")
        gcfg = cfg if isinstance(cfg, GyroDetectionConfig) else GyroDetectionConfig()
        return detect_heelstrikes_gyro(stream.gyro, stream.timestamps,
                                       cfg=gcfg.for_stream(stream))

    acfg = cfg if isinstance(cfg, DetectionConfig) else DetectionConfig()
    return detect_heelstrikes_full(np.asarray(stream.accel, dtype=np.float64),
                                   np.asarray(stream.timestamps),
                                   cfg=acfg.for_stream(stream))


__all__ = ["detect", "detector_name"]
