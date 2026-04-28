# Heel-strike detection pipeline

This document describes the algorithm HITLO_Symmetry uses to detect heel
strikes from shank-mounted IMU signals. It covers the physiologic
justification, each processing stage, the rationale for every threshold, and
the literature that motivated the design choices.

---

## TL;DR

We detect heel strikes in bilateral shank IMU signals using a physics-based
multi-stage pipeline:

```
raw tri-axial accel
   → magnitude |a|
   → 50 Hz lowpass Butterworth, order 4, filtfilt (zero phase delay)
   → differentiate → jerk = |d|a|/dt|
   → z-score normalization
   → strict peak detection (0.7 SD jerk)
   → gap-fill recovery (1.8 SD in anomalously long gaps)
   → cluster candidates (peaks within 0.5 s)
   → within each cluster: scan last-to-first, accept the first peak that is
      (a) above gravity baseline AND
      (b) followed by a stance region (post-peak window near baseline)
   → drop singleton clusters at trial edges
   → trim 3 s from each end
   → filter physiologically-implausible strides (< 0.3 s or > 3.0 s)
```

The result is a timestamped list of heel-strike events per leg, from which
step times and the gait symmetry index are computed.

---

## Why this matters

For a passive ankle exoskeleton optimizer targeting **step-time symmetry** as
its cost, heel-strike detection is the most critical component. Every timing
measurement downstream, every symmetry index, every Bayesian optimization
suggestion — all of it flows from the list of detected events.

Detection errors have directional consequences:

| Error type | Effect on symmetry index |
|---|---|
| False positive (extra heel strike) | Creates a fake short step, biases SI |
| False negative (missed heel strike) | Merges two real strides into one long one |
| Temporal jitter (ms-scale timing error) | Adds noise to step times |
| Wrong-foot label | Catastrophic — inverts the signed SI |

For stroke rehab where the true symmetry may be 15-25%, a detection bias of a
few percent can swamp the signal. **The pipeline's job is to get detection
right, not fast.** We willingly trade compute for specificity.

---

## Physiologic foundation

The entire algorithm rests on two observations about what heel strike
actually *is* mechanically.

### Observation 1: heel strike is an impact

When the foot makes contact with the ground, the shank decelerates suddenly.
The IMU, rigidly coupled to the shank, records a transient acceleration
spike that is large compared to both (a) gravity alone and (b) the
accelerations of swing phase.

Because the shank is moving through 3D space during gait (swinging forward,
rotating), we use the **tri-axial magnitude** `|a| = √(x² + y² + z²)` rather
than any single axis. Magnitude is invariant to sensor orientation on the
shank, which matters in our setup because Coban wrapping doesn't guarantee a
perfectly consistent mounting between sessions.

Quantitatively: `|a|` during stance ≈ 1 g (gravity alone, since the shank
is quasi-stationary). Impact spikes during heel strike reach 2–5 g typically
for walking speeds around 0.8–1.3 m/s.

### Observation 2: stance produces a quasi-stationary shank

Immediately after heel strike, the foot is planted and bears the body's
weight. The shank is rotating slowly if at all (mid-stance it is nearly
vertical and essentially static). The accelerometer reads mostly gravity.

Therefore, in the 100–300 ms after a true heel strike, the magnitude signal
should be close to its baseline value (≈ 1 g). Any candidate peak *not*
followed by such a quasi-static window is, by physiology, not a heel strike.

These two observations — *heel strike is an impact above baseline* and
*heel strike is followed by quasi-stationary stance* — are the two filters
that distinguish real heel strikes from other jerk peaks in the gait cycle
(toe-off, mid-swing transients, etc.).

---

## Pipeline stages

### Stage 1 — Compute tri-axial magnitude

```python
magnitude = √(x² + y² + z²)
```

**Why magnitude and not a single axis?** Three reasons:

1. **Orientation invariance.** If the Coban wrap shifts between trials and
   the x/y/z axes rotate relative to the shank, magnitude is unchanged.
2. **Sensitivity.** Heel-strike energy distributes across all three axes
   depending on shank angle, walking speed, and stride mechanics. Projecting
   to any single axis throws away information.
3. **Physical interpretability.** `|a|` near baseline = quasi-stationary;
   `|a|` spiking = impact. This lets us write physiology-grounded thresholds.

### Stage 2 — Lowpass filter, then differentiate, then z-score

#### Lowpass filter (applied to magnitude FIRST)

```python
b, a = butter(4, 50 Hz / Nyquist, btype='low')
magnitude_smooth = filtfilt(b, a, magnitude)
```

**Why filter before differentiating?** Differentiation amplifies high-
frequency noise (a small-amplitude high-frequency wiggle becomes a large-
amplitude jerk spike after `d/dt`). The textbook approach is to remove that
noise *before* differentiating so the derivative is computed from a clean
signal. We follow that convention.

This was changed from earlier versions of the pipeline. Empirical
validation history is documented below ("Filter-ordering validation").

**Why 50 Hz cutoff?** Heel-strike impact energy concentrates between roughly
5 Hz and 30 Hz (sharp transients lasting 30–80 ms). A cutoff at 50 Hz sits
well above that band, so the filter acts as **light noise cleanup** rather
than reshaping the impact itself. Lower cutoffs (15–30 Hz) cut into the
impact band and visibly distort the impact peak: filtered magnitude shows a
~25% lower peak height and ~doubled peak width compared to raw magnitude.
That distortion propagates into shifted jerk peaks and degraded detection
specificity.

Sheerin et al. (2019, *Gait & Posture* 67:12-24) reviewed filter conventions
in tibial-impact IMU literature and reported that lowpass cutoffs
"between 40 Hz and 100 Hz" are typical, with empirical analyses indicating
that 99% of tibial acceleration spectral power during running falls below
60 Hz. The 50 Hz cutoff used here sits at the lower end of this range.
Note that Sheerin's review concerns peak-amplitude measurement on raw
acceleration; our pipeline differentiates the magnitude signal afterward,
so the appropriateness of the cutoff for our specific use case was
validated empirically (see "Filter-ordering validation" below).

The Polar H10 sensor's electronic noise floor sits well above 50 Hz, so
content above that cutoff is essentially garbage worth removing. Below
50 Hz, the cutoff preserves both the gait fundamentals (~1 Hz) and the
impact harmonics (out to ~30 Hz).

**Why Butterworth?** Butterworth filters have a maximally flat passband (no
ripple), preserving the heel-strike peak shape faithfully. Alternatives
like Chebyshev would give sharper rolloff but introduce ripples that
distort the impact shape.

**Why order 4?** Each order roughly doubles the rolloff rate (order 4 is
~24 dB/octave, which means frequencies above 50 Hz are suppressed by at
least a factor of 16 per doubling). Order 2 is too gentle; order 8+ risks
numerical instability and transient ringing. Order 4 is the biomechanics
standard.

**Why filtfilt (not filter)?** A normal IIR filter introduces a frequency-
dependent phase delay — filtered peaks land slightly *after* raw peaks.
Since we're doing timing-based analysis (step times in milliseconds), delay
is unacceptable. `filtfilt` runs the filter forward then backward, canceling
the delay exactly. The result is zero-phase; the effective order doubles
(so our order-4 gives order-8 rolloff magnitude-wise), which is a bonus.

**Caveat: filtfilt is non-causal.** Because filtfilt uses future samples
(it runs the filter backward as well as forward), this pipeline is **offline
only**. For real-time deployment, this stage would need to be replaced with
a causal FIR filter with documented group delay (and the timestamps
compensated accordingly).

#### Jerk (differentiate the smoothed magnitude)

```python
jerk = |d(magnitude_smooth)/dt|
```

**Why jerk (rate of change of magnitude) instead of raw magnitude?** Jerk
emphasizes *sudden transitions* and suppresses slow ones:

- Heel strike: sharp transition from swing (changing `|a|`) to stance
  (flat `|a|`) — produces a huge jerk spike
- Gravity reorientation during swing: slow change in `|a|` — small jerk
- Baseline drift, sensor warming: very slow — essentially no jerk

Using jerk rather than raw magnitude makes the detector look for *events*
(brief, localized changes) rather than *levels* (absolute amplitudes),
which is exactly what we want for timing individual heel strikes.

Zhou et al. (2016, *Sensors* 16(10):1634) demonstrated this approach
on a shank-mounted IMU, computing jerk as the magnitude of the
acceleration derivative and applying peak heuristics to detect heel
strikes and toe-offs across level-ground, ascending-stair, and
descending-stair walking with F1 scores >0.98 in healthy subjects.
Our pipeline follows this fundamental approach with several extensions
described below (z-score normalization, two-pass detection, cluster
keep-last, and stance verification).

#### Z-score normalization

```python
jerk_z = (jerk - mean) / std
```

**Why z-score?** Thresholds expressed in units of standard deviations above
the noise floor are portable across sessions, sensors, and subjects. A 0.7
SD peak is always meaningful regardless of absolute jerk magnitude, which
varies with participant (stride impact force) and hardware (sensor
calibration drift).

This is mathematically equivalent to an adaptive threshold that
auto-calibrates to each trial's baseline noise level.

### Stage 3 — Two-pass peak detection

Peak detection on the z-scored jerk signal proceeds in two passes.

#### Pass 1 (strict)

```python
strict_peaks = find_peaks(jerk_z, height=0.7, distance=20 samples)
```

Finds all local maxima above 0.7 standard deviations of jerk, separated by
at least 100 ms.

**Why 0.7 SD (and not 2.5 SD as in earlier versions)?** We *want* to cast a
wide net here. False positives are tolerable because the cluster-keep-last
stage later will filter them out physiologically. False negatives are
expensive because every missed heel strike distorts the step-time pattern.
Strategy: *over-detect now, filter aggressively later.*

**Why distance=100 ms?** No real human walking stride produces two heel
strikes less than 100 ms apart. This prevents the peak detector from
double-counting a single peak that has a small notch at its apex.

#### Pass 2 (gap-fill recovery)

After strict detection, we compute the median interval between detected
peaks. If we observe a gap that is greater than 1.7× the median, there
probably was a real heel strike in that window whose jerk happened to fall
below the strict threshold. We search that window with a higher-threshold
fallback:

```python
recovered = find_peaks(jerk_z[gap_start:gap_end],
                       height=1.8, distance=20 samples)
```

If nothing above 1.8 SD exists in the gap, we leave it alone. We do **not
interpolate** or fabricate events — only evidence-based detection.

**Why 1.7× median?** In rhythmic walking, stride-to-stride variation is
typically <10–15%. A gap 70% longer than median is a strong anomaly signal.

**Why 1.8 SD recovery threshold?** Higher than the strict pass (0.7 SD)
because we're already committing a lower specificity by looking in
anomalously long gaps — we want high confidence that any recovered peak
is genuine.

**Why not just interpolate?** An older pipeline version fabricated events
at the midpoint of long gaps. This was bug-bait: fabricated events create
*perfectly symmetric* fake step times (midpoint → equal halves), which
artificially pulls the symmetry index toward zero. For an optimizer trying
to minimize symmetry, that's catastrophic — it would converge toward
parameter settings that *cause* detection failures rather than improve
gait.

### Stage 4 — Cluster candidates

After the two passes, we have a list of candidate peaks with high
sensitivity (very few real heel strikes missed) and low specificity (many
other gait events also present — toe-off, mid-swing transients, post-impact
tissue ringing).

The cluster stage exploits the physiology: **one gait cycle = one heel
strike, but many jerk peaks.** Group peaks that are within 0.5 s of a
neighbor into clusters.

**Why 0.5 s?** This must be long enough to gather all peaks that belong to
one gait cycle (heel strike + post-impact wobble + toe-off-related peaks)
but short enough that two consecutive heel strikes don't accidentally fall
into the same cluster.

Earlier versions used 0.65 s, which was wide enough to span heel strike +
toe-off in the same cluster (the original design). However, at higher
cadences (stride times below ~1.3 s, common in healthy walking and
voluntary asymmetric trials), 0.65 s was wide enough to span *consecutive
heel strikes*, merging two gait cycles into one cluster and losing strides.
Tightening to 0.5 s preserves the original design intent (heel-strike +
post-impact peaks cluster together) while reliably keeping consecutive
heel strikes in separate clusters across the full cadence range we observe.

### Stage 5 — The two physiologic filters

Within each cluster, scan from the **last peak backwards**. The first peak
that satisfies both conditions below is the heel strike; all others in the
cluster are rejected.

#### Filter A: Above baseline

```python
baseline = median(|a|)             # ≈ 1 g across the whole trial
if magnitude[peak_index] < baseline:
    skip this peak
```

**Why:** During mid-swing, the shank can briefly enter near-free-fall
(acceleration magnitude drops well below gravity). The rapid transition
into and out of that trough produces jerk peaks. But these are physics
artifacts, not impacts. A real heel strike has `|a|` *above* baseline.

Using the trial's median `|a|` as baseline is robust to outliers: impact
spikes and occasional gravity-aligned stance phases don't distort the
central value.

#### Filter B: Followed by stance

```python
post_peak_window = magnitude[peak_index + 0.10 s : peak_index + 0.30 s]
mad = mean(|post_peak_window - baseline|)
if mad > 0.15 × baseline:
    skip this peak
```

**Why:** A real heel strike is *immediately followed by stance*: the foot
is planted, the shank is quasi-stationary, and the accelerometer reads
mostly gravity. In a 200 ms window starting 100 ms after peak (skipping
impact ring-down), the magnitude should stay close to baseline.

We compute mean absolute deviation (MAD) from baseline across that window.
If MAD exceeds 15% of baseline (~150 mg on a 1000 mg baseline), the shank
is *still moving* — this was not a heel strike, it was probably a toe-off
or a mid-swing transient. Reject.

**Why MAD and not peak-to-peak or std?** MAD is robust:

- Peak-to-peak would flag a stance window because of one stray sample
- std is sensitive to the tails of the distribution
- MAD averages "how far from baseline" across the window, which is exactly
  what the physiologic question is asking

**Why 15% tolerance (not tighter)?** Stroke gait in particular has
compensatory strategies during stance (small lateral adjustments, postural
reactions) that produce modest but real variability. 15% captures the
"essentially stationary" regime while tolerating normal gait-by-gait
variation.

#### Why scan last-to-first?

Within one gait cycle, the event sequence is:

```
heel-off → toe-off → (swing, possible transients) → heel-strike → stance
```

Only heel-strike is followed by stance. Everything earlier in the cycle
has more motion following (swing continuing). So the natural rule is:
walk backwards from the last peak in the cluster; the first one that
passes both filters is heel-strike.

If no peak in a cluster passes both filters, the cluster emits no event.
We never fabricate. (This can happen if a cluster caught the start of a
partial cycle at the trial edge, or in cases of extreme gait irregularity.)

### Stage 6 — Edge-singleton rejection

Singletons at the first and last cluster of a trial are often partial
cycles — the subject started or stopped walking mid-stride, or the trial
boundary clipped the signal. Their timing is unreliable. We drop them.

Singletons in the middle of the trial are usually real: they pass both
physiologic filters, and we keep them.

### Stage 7 — Steady-state trim

```python
drop any heel strike within 3 s of trial start or end
```

**Why:** Ramp-up (starting from standstill) and ramp-down (decelerating to
stop) produce systematically different shank mechanics than steady-state
walking. Weaker impacts, asymmetric timing, non-representative gait. This
biases the symmetry estimate toward whatever asymmetry the participant
naturally exhibits during start/stop — which is different from their
steady-state walking pattern that the exoskeleton is trying to optimize.

Standard practice in gait literature is 3–5 s trim; we use 3 s.

### Stage 8 — Physiologic plausibility filter

```python
if interval < 0.3 s:     # faster than any real stride
    drop the later heel strike
if interval > 3.0 s:     # slower than any continuous walking
    flag as possible missed detection
```

Remaining heel strikes are paired across L/R to produce step times.

### Stage 9 — Symmetry index

For each left/right heel-strike pair:

```
Right step = time from LEFT heel strike to next RIGHT heel strike
Left step  = time from RIGHT heel strike to next LEFT heel strike

per-stride SI = 2 × (right_step - left_step) / (right_step + left_step) × 100 %
```

Aggregate: mean per-stride SI (signed) across all stride pairs.

**Interpretation:**

| SI | Meaning |
|---|---|
| SI = 0 | Perfectly symmetric |
| SI > 0 | Right step longer → left leg is support-dominant |
| SI < 0 | Left step longer → right leg is support-dominant |
| &#124;SI&#124; < 2% | Near-symmetric (healthy young adult typical) |
| 2–10% | Mild asymmetry |
| 10–20% | Moderate asymmetry |
| > 20% | Severe asymmetry (often chronic stroke) |

The **signed** SI is what we use as the BO cost, because we want the
optimizer to converge to zero (true symmetry). Unsigned SI would allow
the optimizer to get stuck at any point where |asymmetry| = target
regardless of direction, which is not the clinical goal.

---

## Filter-ordering validation

A previous iteration of this pipeline differentiated the magnitude *first*
and then lowpass-filtered the resulting jerk. The current pipeline reverses
that order (filter first, differentiate second), which is the textbook
signal-processing convention.

This change was empirically validated on participant P048 (run-007). Three
filter cutoffs were tested under both orderings:

| Cutoff | OLD (diff-then-filter) | NEW (filter-then-diff) |
|--------|-----------------------|------------------------|
| 15 Hz  | 22 / 22 accepted, SI std 10.4% | 20 / 22 accepted, SI std 20.9% |
| 30 Hz  | (not tested in original pipeline) | 20 / 22 accepted, SI std 16.3% |
| 45 Hz  | 21 / 22 accepted, SI std ~10% | 21 / 22 accepted, SI std ~10% |

At the original 15 Hz cutoff, filter-first dropped 2 real heel strikes and
doubled per-stride SI variance. Inspection showed the 15 Hz lowpass smearing
the impact peak (~25% reduction in peak height, ~doubled peak width), so the
subsequent differentiation no longer produced a clean spike.

At a 45–50 Hz cutoff, the lowpass acts well above the impact band and barely
affects the magnitude signal in the impact region. At that point ordering
becomes irrelevant — both pipelines produce nearly identical jerk signals
and detection counts. The current pipeline uses **50 Hz** as its default
cutoff with filter-first ordering, capturing the textbook convention while
preserving the impact fidelity that motivated the original design.

The validation script (`compare_filter_order.py`) lives outside the main
repository in the author's local-tools directory; it produces a 6-panel
figure including a zoomed comparison of raw vs. filtered magnitude
through a single impact, which is the most direct visualization of whether
a given cutoff is destroying impact information.

---

## Why NOT other approaches

### Why not train an ML classifier?

A deep learning model on shank IMU data could outperform rule-based detection
if given enough training data. For a single-institution preliminary study
with ~20 subjects, it is not feasible to collect enough labeled data to
outperform a well-tuned rule-based detector, especially because stroke gait
presentations are heterogeneous.

Rule-based detection has the additional benefit of being
**physiologically interpretable** — every threshold corresponds to a
physical quantity (impact, stance, stride rate). Reviewers, clinicians,
and future students can understand and tune it without ML background.

### Why not template matching / DTW?

Template-matching approaches (e.g. Voisard et al. 2024, which uses
autocorrelation to identify a reference stride pattern and then
multiparametric Dynamic Time Warping to annotate gait events) are
filter-tolerant by design and achieve excellent accuracy on healthy
subjects (F1 = 100%, median timing error 8 ms in their data). However,
they require a coherent stride pattern to bootstrap the template. Stride-
to-stride variability in stroke gait can degrade the autocorrelation
peak, and template matching adds a tunable component (the template
itself) that must be characterized. For a real-time BO loop where each
60-second trial must be processed independently and reproducibly,
threshold-based detection on jerk peaks is simpler to inspect and
debug, even at some accuracy cost.

### Why not raw accelerometer peak detection (like the old sternum pipeline)?

Previous work in our lab used a single sternum-mounted sensor with peak
detection on the forward-aft axis and an assumed left-right peak
alternation. This approach has three fundamental limitations:

1. **Cannot distinguish L from R.** With one sensor at midline, there is
   no signal difference between left and right heel strikes. The pipeline
   assumes alternation, which fails for irregular gait or missed events.
2. **Uses a weak threshold** (signal mean). Catches arm-swing and torso
   rotation artifacts.
3. **Fabricates missing events.** The "halve long intervals" rule
   artificially improves the symmetry index by construction — a critical
   flaw for an optimizer targeting symmetry.

The current two-sensor shank-mounted pipeline directly addresses all three
limitations.

---

## Validation

The pipeline was validated on participant P048 (healthy, voluntarily
asymmetric gait) across 7 trials. On run-007 (volitional right-step-longer
walk), the pipeline detected:

- LEFT sensor: ~88 candidates → 21 heel strikes (after cluster + filter)
- RIGHT sensor: ~87 candidates → 22 heel strikes
- After 3 s trim: 18 + 18 heel strikes
- Right step mean: ~0.98 s; Left step mean: ~0.84 s
- **SI ≈ +14%** (right step ~14% longer, matching participant intent)

Filter-ordering and cutoff choices were validated against the original
diff-then-filter pipeline at 15 Hz; see "Filter-ordering validation" above.

---

## References

**Primary methodological precedents:**

Zhou H, Ji N, Samuel OW, Cao Y, Zhao Z, Chen S, Li G (2016).
Towards Real-Time Detection of Gait Events on Different Terrains
Using Time-Frequency Analysis and Peak Heuristics Algorithm.
*Sensors* 16(10):1634. doi:10.3390/s16101634. PMID: 27706086.
**Cited for:** Jerk-based heel-strike and toe-off detection from
shank-mounted accelerometer signals; peak heuristics with
adaptive thresholds and refractory windows. F1 > 0.98 for
heel-strike detection on level ground, ascending stairs, and
descending stairs in healthy subjects.

Sheerin KR, Reid D, Besier TF (2019). The measurement of tibial
acceleration in runners — a review of the factors that can affect
tibial acceleration during running and evidence-based guidelines
for its use. *Gait & Posture* 67:12-24.
doi:10.1016/j.gaitpost.2018.09.017. PMID: 30248663.
**Cited for:** Conventional lowpass filter cutoff range
(40-100 Hz, modal value 60 Hz) for tibial-impact IMU signal
preprocessing. Empirical observation that 99% of tibial
acceleration spectral power during locomotion falls below 60 Hz.

**Comparative methods (alternative approaches we considered but did not adopt):**

Voisard C, de l'Escalopier N, Ricard D, Oudre L (2024). Automatic
gait events detection with inertial measurement units: healthy
subjects and moderate to severe impaired patients.
*J NeuroEng Rehabil* 21:104. doi:10.1186/s12984-024-01405-x.
PMID: 38890696.
**Cited for:** Template-based gait event detection using
autocorrelation and multiparametric Dynamic Time Warping on
filtered acceleration signals. Excellent accuracy (F1=100%
healthy, median 8 ms timing error) but requires bootstrapping a
stride template from coherent input data. Different methodology
from our threshold-based approach.

Trojaniello D, Cereatti A, Pelosin E, Avanzino L, Mirelman A,
Hausdorff JM, Della Croce U (2014). Estimation of step-by-step
spatio-temporal parameters of normal and impaired gait using
shank-mounted magneto-inertial sensors: application to elderly,
hemiparetic, parkinsonian and choreic gait. *J NeuroEng Rehabil*
11:152. doi:10.1186/1743-0003-11-152. PMID: 25388296.
**Cited for:** Shank-mounted IMU gait event detection in
pathological gait via medial-lateral angular velocity features
(an alternative kinematic detection approach to our impact-based
approach).
