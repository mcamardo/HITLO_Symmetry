"""
hitlo.palette — one place that decides what colour a leg is.

Left is red, right is green, following the navigation-light convention. Import
from here rather than writing a colour literal into a plot: the repo had left
as steelblue in one figure, blue in another and tomato-vs-darkred in a third,
so the same leg changed colour between two panels of the same report.

    from hitlo.palette import LEFT, RIGHT, LEFT_DK, RIGHT_DK

WHY THESE PARTICULAR RED AND GREEN
----------------------------------
Red against green is the classic colour-vision trap, so the two steps were
chosen by measurement rather than by eye. Separation is OKLab dE x100, under
Vienot 1999 dichromat simulation, against the cream figure background:

                       normal   deuter   protan   contrast on cream
    #b5121b / #2e7d32    28.3     26.4     18.3    6.09:1 / 4.56:1   <- chosen
    #b5121b / #1b6b34    27.4     31.9     12.3    6.09:1 / 5.84:1
    #c0392b / #1b6b34    26.3     35.1      5.2    4.84:1 / 5.84:1   fails protan
    #b5121b / #4cae4f    35.1     17.0     33.8    6.09:1 / 2.50:1   green too light

The chosen pair maximises the WORST case across both dichromacies rather than
either one alone, and both steps clear 4.5:1 against the background, which the
previous green (2.50:1) did not — it was too light to read as a thin line.

Colour still never carries identity on its own. Left and right go in separate
panels wherever possible; where they share an axis, the right leg is dashed.
"""

# Legs. Left = red, right = green.
LEFT = "#b5121b"
RIGHT = "#2e7d32"

# Darker steps, for markers sitting on top of their own trace.
LEFT_DK = "#7a0c12"
RIGHT_DK = "#1b5e20"

# Lighter steps, for fills and confidence bands.
LEFT_LT = "#e8a5a8"
RIGHT_LT = "#a8d3aa"

# Non-leg ink. Grid and axes stay recessive; rejected marks are grey so they
# read as absent rather than as a third leg.
INK = "#12131a"
MUTE = "#6b6f76"
GRID = "#e2e4e8"
SURFACE = "#faf0e6"

# Detector states and annotations, kept out of the leg hues so nothing shaded
# or thresholded can be mistaken for a leg. This matters more than it sounds:
# the cluster shading used to be salmon and limegreen, which was harmless while
# the legs were blue and orange and became unreadable the moment left turned
# red -- a pink band behind a red trace reads as part of the trace.
STRICT = "#1f6f8b"
RECOVERY = "#c77b1f"
REJECT = "#9aa0a6"
CLUSTER_MULTI = "#7e57c2"    # violet: several peaks collapsed into one strike
CLUSTER_SINGLE = "#00838f"   # teal: a lone peak
TRIM = "#9aa0a6"


def leg(side: str, dark: bool = False, light: bool = False) -> str:
    """Colour for 'left' or 'right'. Raises rather than guessing.

    A silent default here would paint an unknown side the same as a known one,
    which is exactly the failure this module exists to prevent.
    """
    s = str(side).strip().lower()
    if s not in ("left", "right", "l", "r"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    is_left = s in ("left", "l")
    if dark:
        return LEFT_DK if is_left else RIGHT_DK
    if light:
        return LEFT_LT if is_left else RIGHT_LT
    return LEFT if is_left else RIGHT


def linestyle(side: str) -> str:
    """Secondary, non-colour cue for when both legs share one axis."""
    return "-" if str(side).strip().lower() in ("left", "l") else "--"
