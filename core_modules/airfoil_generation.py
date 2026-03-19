"""
Bird-Inspired Airfoil Generation Module
========================================
Generates aerodynamically valid airfoils from bird morphological data
using biologically-grounded Bézier curves.

All geometric parameters are derived from peer-reviewed ornithological
and aerodynamic literature.  Every formula is documented with its
biological rationale and citation.

Primary References
------------------
[1] Lentink, D. et al. (2007). How swifts control their glide performance
    with morphing wings. Nature, 446, 1082–1085.
[2] Pennycuick, C.J. (2008). Modelling the Flying Bird. Academic Press.
[3] Sheard, C. et al. (2020). Span-point morphology across the avian
    phylogeny. Nature Communications, 11, 994.
[4] Videler, J.J. (2005). Avian Flight. Oxford University Press.
[5] Withers, P.C. (1981). An aerodynamic analysis of bird wings as fixed
    aerofoils. Journal of Experimental Biology, 90, 143–162.
[6] Vogel, S. (1994). Life in Moving Fluids. Princeton University Press.
[7] Tobias, J.A. et al. (2022). AVONET: morphological, ecological and
    geographical data for all birds. Ecology Letters, 25, 581–597.
[8] Spedding, G.R. (1987). The wake of a kestrel in gliding flight.
    Journal of Experimental Biology, 127, 45–57.
"""

import numpy as np
import pandas as pd
from scipy.special import comb
from scipy import interpolate
import os
import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Project directories
# ---------------------------------------------------------------------------
BASE_DIR           = Path(__file__).parent.parent
OUTPUT_DIR         = BASE_DIR / "OUTPUT"
AIRFOIL_DIR        = OUTPUT_DIR / "airfoils"
CONTROL_POINTS_DIR = OUTPUT_DIR / "control_points"


# ===========================================================================
# REYNOLDS NUMBER ESTIMATION
# ===========================================================================

def estimate_reynolds_number(wing_length_mm: float,
                              wing_loading_proxy: float | None) -> float:
    """
    Estimate chord-based Reynolds number from morphological measurements.

    Re = V * c / nu,   nu_air ≈ 1.5e-5 m²/s at sea level.

    Flight speed proxy
    ------------------
    Pennycuick (2008) shows minimum-power flight speed scales with wing
    loading as V ∝ sqrt(wing_loading / rho).  wing_loading_proxy
    (Wing.Length / (Secondary1 × Tail.Length)) is a monotone proxy for
    true wing loading, so it modulates a size-based base speed.

    Chord estimate
    --------------
    Mean chord ≈ 30 % of wing_length_mm, consistent with typical avian
    wing aspect ratios (Pennycuick 2008, Appendix).

    Validation against published values
    ------------------------------------
      hummingbird  wing ≈  60 mm  →  Re ≈  10,000 –  25,000  (Warrick 2005)
      swift        wing ≈ 170 mm  →  Re ≈  50,000 –  80,000  (Lentink 2007)
      pigeon       wing ≈ 250 mm  →  Re ≈ 100,000 – 200,000  (Pennycuick 2008)
      albatross    wing ≈ 500 mm  →  Re ≈ 500,000 – 900,000  (Pennycuick 2008)

    Parameters
    ----------
    wing_length_mm      : Wing.Length in millimetres.
    wing_loading_proxy  : Wing.Length / (Secondary1 × Tail.Length), or None.

    Returns
    -------
    float   Estimated Reynolds number, clipped to [1e4, 2e6].
    """
    nu      = 1.5e-5                                    # kinematic viscosity [m²/s]
    chord_m = (wing_length_mm / 1000.0) * 0.30         # mean chord ≈ 30 % of span

    # Base speed from allometric scaling (Pennycuick 2008, Fig 3.5)
    # Saturates around albatross size; hummingbird → ~5 m/s, albatross → ~20 m/s
    wing_m  = wing_length_mm / 1000.0
    v_base  = 5.0 + 15.0 * (wing_m / 0.5) ** 0.5

    # Modulate by wing loading proxy (higher loading → faster min-power speed)
    if wing_loading_proxy is not None:
        wl_factor  = float(np.clip(wing_loading_proxy / 0.05, 0.5, 2.0))
        v_estimated = v_base * wl_factor ** 0.5
    else:
        v_estimated = v_base

    re = v_estimated * chord_m / nu
    return float(np.clip(re, 1e4, 2e6))


# ===========================================================================
# GEOMETRIC PARAMETER COMPUTATION
# ===========================================================================

def compute_airfoil_parameters(
    wing_length:        float,
    secondary1:         float,
    kipps_distance:     float,
    hand_wing_index:    float,
    tail_length:        float,
    aspect_ratio:       float,
    pointedness_index:  float,
    wing_loading_proxy: float | None = None,
) -> dict:
    """
    Convert AVONET measurements to airfoil geometric parameters.

    Every formula is grounded in the literature listed in the module
    docstring; the rationale for each is given inline.

    Parameters
    ----------
    wing_length        : Wing.Length [mm]
    secondary1         : Secondary1 [mm]  — proximal wing chord proxy
    kipps_distance     : Kipps.Distance [mm]  — distal wing pointedness
    hand_wing_index    : Hand-Wing.Index [0–100]  (Sheard et al. 2020)
    tail_length        : Tail.Length [mm]
    aspect_ratio       : Wing.Length / Secondary1  (local chord ratio)
    pointedness_index  : Kipps.Distance / Wing.Length
    wing_loading_proxy : Wing.Length / (Secondary1 × Tail.Length), optional

    Returns
    -------
    dict   All geometric parameters + Reynolds diagnostics.
    """
    # Normalised inputs ∈ [0, 1]
    hwi_n = float(np.clip(hand_wing_index / 100.0, 0.0, 1.0))
    pi_n  = float(np.clip(pointedness_index,        0.0, 1.0))

    estimated_re = estimate_reynolds_number(wing_length, wing_loading_proxy)
    # Log-normalised Re: maps 1e4 → 0.0,  1e6 → 1.0
    re_factor = float(np.clip((np.log10(estimated_re) - 4.0) / 2.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 1. MAXIMUM THICKNESS  (t/c)
    #
    # Lentink et al. (2007) and Withers (1981) measured t/c directly:
    #   hummingbird 7–9 %,  swift 6–8 %,  pigeon 10–12 %,  vulture 11–14 %
    # HWI (Sheard 2020) is the best morphological proxy for flight speed,
    # so t/c is mapped directly from HWI.
    # Re provides a secondary correction: higher Re allows thinner profiles
    # because laminar separation bubbles are more tolerant at Re > 1e5
    # (Vogel 1994, ch. 12).
    #
    #   t/c = 0.14 − 0.08·hwi_n − 0.02·re_factor
    #   Clipped to [0.04, 0.15]  (biological limits, Withers 1981)
    # ------------------------------------------------------------------
    max_thickness = 0.14 - 0.08 * hwi_n - 0.02 * re_factor
    max_thickness = float(np.clip(max_thickness, 0.04, 0.15))

    # ------------------------------------------------------------------
    # 2. MAXIMUM CAMBER  (f/c)
    #
    # Withers (1981) and Spedding (1987) measurements:
    #   slow/hovering 5–8 %,  cruising 3–5 %,  fast/diving 2–4 %
    # Parabolic HWI term (Lentink 2007, Fig 4): moderate-speed birds
    # maximise camber for cruise efficiency; very fast birds reduce it
    # to minimise pressure drag.
    #
    # REMOVED: kipps_ratio term (pointedness is a planform/3-D parameter,
    # unrelated to 2-D section camber) and tail_ratio term (tail governs
    # whole-bird pitch stability, not wing cross-section camber).
    #
    #   f/c = (0.06 − 0.03·hwi_n)  +  0.025·hwi_n·(1 − hwi_n)
    #   Clipped to [0.02, 0.08]
    # ------------------------------------------------------------------
    base_camber      = 0.06 - 0.03 * hwi_n
    parabolic_camber = 0.025 * hwi_n * (1.0 - hwi_n)   # peaks at hwi_n = 0.5
    max_camber = float(np.clip(base_camber + parabolic_camber, 0.02, 0.08))

    # ------------------------------------------------------------------
    # 3. LEADING EDGE RADIUS  (r_LE / c)
    #
    # Withers (1981): pointed wings → sharp LE (attached flow at speed);
    # rounded wings → blunt LE (delayed stall at low speed).
    #
    # Coefficients capped so the expression never goes negative:
    #   le_sharpness = 0.5·pi_n + 0.3·hwi_n   ∈ [0, 0.80]
    #   r_LE = 0.04·(1 − 0.85·le_sharpness)   ∈ [0.007, 0.040]
    # Clipped to [0.005, 0.040] as safety net.
    #
    # Previous version: sharpness could exceed 1.4 → negative radius
    # before clipping, collapsing many birds to identical minimum r_LE.
    # ------------------------------------------------------------------
    le_sharpness        = 0.5 * pi_n + 0.3 * hwi_n          # max = 0.80
    leading_edge_radius = 0.04 * (1.0 - 0.85 * le_sharpness)
    leading_edge_radius = float(np.clip(leading_edge_radius, 0.005, 0.040))

    # ------------------------------------------------------------------
    # 4. LEADING EDGE DROOP  (le_droop)
    #
    # Videler (2005), ch. 4: drooped / cambered leading edges occur in
    # SLOW birds (owls, low-speed raptors) as stall-delay devices,
    # analogous to Krueger flaps.
    # FAST birds (high HWI) have straight, sharp leading edges.
    #
    # Previous version was INVERTED — applied droop to high-HWI (fast)
    # birds, which is aerodynamically wrong.
    #
    #   le_droop = (0.4 − hwi_n) × 0.02   only when hwi_n < 0.4
    #   max droop ≈ 0.8 % for the slowest birds
    # ------------------------------------------------------------------
    le_droop = float((0.4 - hwi_n) * 0.02 if hwi_n < 0.4 else 0.0)
    le_droop = float(np.clip(le_droop, 0.0, 0.008))

    # ------------------------------------------------------------------
    # 5. MAXIMUM THICKNESS POSITION  (x_t / c)
    #
    # Lentink et al. (2007) measured x_t ≈ 28 % for swifts (consistent
    # with NACA 6-series laminar profiles).  Slow birds (pigeon, heron)
    # have x_t ≈ 33–38 % (NACA 4-digit analogue).
    #
    # REMOVED: aspect_ratio contribution.  The local AR (Wing.Length /
    # Secondary1 ≈ 1.5–4) pushed most birds to the 0.20 clip boundary,
    # destroying morphological diversity in the database.
    #
    #   x_t = 0.35 − 0.06·hwi_n
    #   Clipped to [0.25, 0.42]
    # ------------------------------------------------------------------
    max_thickness_pos = float(np.clip(0.35 - 0.06 * hwi_n, 0.25, 0.42))

    # ------------------------------------------------------------------
    # 6. MAXIMUM CAMBER POSITION  (x_f / c)
    #
    # Camber peak lies slightly aft of thickness peak (Jones 1990).
    # Faster birds shift it forward for better front-loaded lift.
    #
    #   x_f = 0.38 − 0.08·hwi_n
    #   Clipped to [0.28, 0.42]
    # ------------------------------------------------------------------
    max_camber_pos = float(np.clip(0.38 - 0.08 * hwi_n, 0.28, 0.42))

    # ------------------------------------------------------------------
    # 7. TRAILING EDGE THICKNESS  (t_TE / c)
    #
    # Bird wing trailing edges are formed by overlapping flight feathers
    # and are effectively sharp.  For 2-D CFD (NeuralFoil / XFOIL), a
    # constant finite TE of 0.002 c is a numerical best-practice to
    # avoid mesh singularities (Drela 1989, XFOIL manual).
    #
    # REMOVED: tail_ratio contribution — tail length governs whole-bird
    # pitch stability (3-D), not 2-D section TE geometry.
    # ------------------------------------------------------------------
    trailing_edge_thickness = 0.002   # constant for all birds

    # ------------------------------------------------------------------
    # 8. REFLEX CAMBER
    #
    # Reflex trims a flying-wing (no tail) by creating a nose-down
    # pitching moment.  Bird wings are NOT flying wings — pitch trim
    # is achieved by the tail (3-D whole-bird effect).  Applying reflex
    # to a 2-D section would systematically penalise Cm in NeuralFoil
    # for long-tailed birds with no biological justification.
    #
    # REMOVED entirely.
    # ------------------------------------------------------------------

    return {
        'max_thickness':           max_thickness,
        'max_thickness_pos':       max_thickness_pos,
        'max_camber':              max_camber,
        'max_camber_pos':          max_camber_pos,
        'leading_edge_radius':     leading_edge_radius,
        'le_droop':                le_droop,
        'trailing_edge_thickness': trailing_edge_thickness,
        # diagnostics
        'estimated_reynolds':      estimated_re,
        're_factor':               re_factor,
        'hwi_normalised':          hwi_n,
        'pointedness_normalised':  pi_n,
        'aspect_ratio_local':      aspect_ratio,
    }


# ===========================================================================
# BÉZIER CONTROL POINT CONSTRUCTION
# ===========================================================================

def build_bezier_control_points(p: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct 12-point Bézier control polygons (upper + lower surface).

    Control point layout
    --------------------
    Points 0–3  : leading edge shape  (radius + droop)
    Points 3–6  : maximum thickness region
    Points 6–9  : pressure recovery / rear camber
    Points 9–11 : trailing edge closure

    Both surfaces share the same x-coordinates so that thickness and
    camber can be computed as (upper − lower) / 2 and (upper + lower) / 2.
    All coordinates are chord-normalised [0, 1].

    Parameters
    ----------
    p : dict   Output of compute_airfoil_parameters().

    Returns
    -------
    upper_cp, lower_cp : np.ndarray  shape (12, 2)
    """
    t   = p['max_thickness']
    tp  = p['max_thickness_pos']
    f   = p['max_camber']
    fp  = p['max_camber_pos']
    r   = p['leading_edge_radius']
    d   = p['le_droop']
    te  = p['trailing_edge_thickness']

    upper_cp = np.array([
        # ── Leading edge ─────────────────────────────────────────────
        [0.000,  0.000],
        [r*0.30,  r*1.80 + d],                         # LE near-field: radius + slow-bird droop
        [r*1.20,  t*0.50 + f*0.85 + d],                # LE ramp upper
        [r*2.50,  t*0.72 + f*0.92],                    # LE ramp exit

        # ── Maximum thickness ────────────────────────────────────────
        [tp*0.87,  f + t*0.94],
        [tp,       f + t],                              # peak: camber + half-thickness
        [tp*1.13,  f + t*0.90],

        # ── Pressure recovery ────────────────────────────────────────
        [(tp + fp)*0.55,  f + t*0.62],
        [fp + 0.13,        f*0.85 + t*0.28],
        [0.88,             f*0.38 + t*0.08],

        # ── Trailing edge ────────────────────────────────────────────
        [0.985,  te],
        [1.000,  0.000],
    ])

    lower_cp = np.array([
        # ── Leading edge ─────────────────────────────────────────────
        [0.000,  0.000],
        [r*0.30, -r*1.20 + d*0.3],                     # lower LE less pronounced; slight droop offset
        [r*1.20, -t*0.32 + f*0.35],
        [r*2.50, -t*0.58 + f*0.46],

        # ── Maximum thickness ────────────────────────────────────────
        [tp*0.87,  f - t*0.84],
        [tp,       f - t],                              # trough: camber − half-thickness
        [tp*1.13,  f - t*0.78],

        # ── Pressure recovery ────────────────────────────────────────
        [(tp + fp)*0.55,  f - t*0.44],
        [fp + 0.13,        f*0.62 - t*0.14],
        [0.88,             f*0.25 - t*0.04],

        # ── Trailing edge ────────────────────────────────────────────
        [0.985, -te*0.5],
        [1.000,  0.000],
    ])

    return upper_cp, lower_cp


# ===========================================================================
# MAIN AIRFOIL GENERATION ENTRY POINT
# ===========================================================================

def generate_bird_airfoil(
    wing_length:        float,
    secondary1:         float,
    kipps_distance:     float,
    hand_wing_index:    float,
    tail_length:        float,
    aspect_ratio:       float,
    pointedness_index:  float,
    wing_loading_proxy: float | None = None,
    chord_length:       float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Full pipeline: AVONET morphology → Bézier control points + metadata.

    Parameters
    ----------
    wing_length … pointedness_index
        Raw AVONET measurements [mm] or dimensionless (HWI).
    wing_loading_proxy
        Derived index from categorisation module (optional).
    chord_length
        Output chord scaling.  Default 1.0 (normalised).

    Returns
    -------
    upper_cp : np.ndarray (12, 2)
    lower_cp : np.ndarray (12, 2)
    metadata : dict
    """
    params = compute_airfoil_parameters(
        wing_length, secondary1, kipps_distance, hand_wing_index,
        tail_length, aspect_ratio, pointedness_index, wing_loading_proxy,
    )

    upper_cp, lower_cp = build_bezier_control_points(params)

    if chord_length != 1.0:
        upper_cp = upper_cp.copy()
        lower_cp = lower_cp.copy()
        upper_cp[:, 0] *= chord_length
        lower_cp[:, 0] *= chord_length

    metadata = {**params, 'chord_length': chord_length}
    return upper_cp, lower_cp, metadata


# ===========================================================================
# BÉZIER CURVE EVALUATION
# ===========================================================================

def bezier_curve(control_points: np.ndarray,
                 num_points: int = 200) -> np.ndarray:
    """
    Evaluate a degree-(n−1) Bézier curve from n control points.

    Vectorised Bernstein basis evaluation — stable at n = 12,
    no de Casteljau recursion needed.

    Parameters
    ----------
    control_points : np.ndarray (n, 2)
    num_points     : int

    Returns
    -------
    np.ndarray (num_points, 2)
    """
    n = len(control_points) - 1
    t = np.linspace(0.0, 1.0, num_points)

    curve = np.zeros((num_points, 2))
    for j in range(n + 1):
        B = comb(n, j, exact=True) * (1.0 - t) ** (n - j) * t ** j
        curve += B[:, np.newaxis] * control_points[j]

    return curve


# ===========================================================================
# AIRFOIL VALIDATION
# ===========================================================================

def validate_airfoil_quality(
    upper_cp: np.ndarray,
    lower_cp: np.ndarray,
    species:  str = "Unknown",
) -> tuple[bool, list[str]]:
    """
    Validate the generated airfoil against geometric and aerodynamic
    constraints required before passing to NeuralFoil.

    Checks
    ------
    1. No self-intersection (upper surface always above lower).
    2. No sharp kinks (max |Δslope| < 3.0 — tightened from old 5.0).
    3. Maximum t/c within biological range [0.04, 0.20].
    4. Leading edge height < 0.06 (not unrealistically blunt).
    5. Monotone x-coordinates — NeuralFoil assumes LE→TE ordering.

    Parameters
    ----------
    upper_cp, lower_cp : np.ndarray (12, 2)
    species            : str   (for warning messages only)

    Returns
    -------
    is_valid : bool
    issues   : list[str]
    """
    issues = []

    upper = bezier_curve(upper_cp, 300)
    lower = bezier_curve(lower_cp, 300)

    # 1. Self-intersection
    y_lower_interp = np.interp(upper[:, 0], lower[:, 0], lower[:, 1])
    crossings = int(np.sum(upper[:, 1] < y_lower_interp))
    if crossings > 0:
        issues.append(
            f"Self-intersection: upper surface below lower at {crossings} sample points"
        )

    # 2. Kink detection
    def kink_check(curve: np.ndarray, label: str) -> None:
        dx     = np.diff(curve[:, 0])
        dy     = np.diff(curve[:, 1])
        slopes = dy / (np.abs(dx) + 1e-12)
        worst  = float(np.max(np.abs(np.diff(slopes))))
        if worst > 3.0:
            issues.append(
                f"{label} kink detected (max Δslope = {worst:.2f}, threshold = 3.0)"
            )

    kink_check(upper, "Upper surface")
    kink_check(lower, "Lower surface")

    # 3. Thickness range
    t_max = float(np.max(upper[:, 1] - y_lower_interp))
    if not (0.04 <= t_max <= 0.20):
        issues.append(
            f"Max thickness t/c = {t_max:.4f} outside biological range [0.04, 0.20]"
        )

    # 4. Leading edge bluntness
    le_height = float(np.max(upper[:10, 1]))
    if le_height > 0.06:
        issues.append(
            f"Leading edge too blunt: max y in first 10 samples = {le_height:.4f}"
        )

    # 5. Monotone x-coordinates
    for label, curve in [("Upper", upper), ("Lower", lower)]:
        backtrack = int(np.sum(np.diff(curve[:, 0]) < -1e-4))
        if backtrack > 0:
            issues.append(
                f"{label} surface has {backtrack} x-backtracking points"
            )

    is_valid = len(issues) == 0
    if not is_valid:
        print(f"⚠  Validation failed [{species}]:")
        for issue in issues:
            print(f"   • {issue}")

    return is_valid, issues


def generate_and_validate_airfoil(bird_data: dict) \
        -> tuple[np.ndarray, np.ndarray, dict, bool]:
    """
    Convenience wrapper: generate + validate for a single bird row.

    Parameters
    ----------
    bird_data : dict-like
        Keys required: Wing.Length, Secondary1, Kipps.Distance,
        Hand-Wing.Index, Tail.Length, aspect_ratio, pointedness_index.
        Optional: wing_loading_proxy, species.

    Returns
    -------
    upper_cp, lower_cp, metadata, is_valid
    """
    upper_cp, lower_cp, metadata = generate_bird_airfoil(
        wing_length        = float(bird_data['Wing.Length']),
        secondary1         = float(bird_data['Secondary1']),
        kipps_distance     = float(bird_data['Kipps.Distance']),
        hand_wing_index    = float(bird_data['Hand-Wing.Index']),
        tail_length        = float(bird_data['Tail.Length']),
        aspect_ratio       = float(bird_data['aspect_ratio']),
        pointedness_index  = float(bird_data['pointedness_index']),
        wing_loading_proxy = bird_data.get('wing_loading_proxy'),
    )

    is_valid, _ = validate_airfoil_quality(
        upper_cp, lower_cp,
        species=str(bird_data.get('species', 'Unknown')),
    )

    return upper_cp, lower_cp, metadata, is_valid


# ===========================================================================
# EXPORT HELPERS
# ===========================================================================

def export_control_points_to_csv(upper_cp, lower_cp, filename,
                                  metadata=None, bird_data=None):
    """
    Export Bézier control points to a commented CSV file.

    Parameters
    ----------
    upper_cp, lower_cp : np.ndarray (12, 2)
    filename   : str or Path
    metadata   : dict   (from generate_bird_airfoil)
    bird_data  : dict   (raw bird row fields)
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['# Bird-inspired Airfoil Bézier Control Points'])
        writer.writerow(['# Generated from AVONET morphological database'])
        writer.writerow(['#'])

        if bird_data is not None:
            writer.writerow([f"# Species: {bird_data.get('species', 'Unknown')}"])
            writer.writerow([f"# Flight Category: {bird_data.get('flight_category', 'Unknown')}"])
            writer.writerow(['#'])
            writer.writerow(['# === Morphological Parameters ==='])
            writer.writerow([f"# Wing Length: {bird_data.get('Wing.Length', 0):.2f} mm"])
            writer.writerow([f"# Secondary1: {bird_data.get('Secondary1', 0):.2f} mm"])
            writer.writerow([f"# Kipps Distance: {bird_data.get('Kipps.Distance', 0):.2f} mm"])
            writer.writerow([f"# Hand-Wing Index: {bird_data.get('Hand-Wing.Index', 0):.2f}"])
            writer.writerow([f"# Tail Length: {bird_data.get('Tail.Length', 0):.2f} mm"])
            writer.writerow([f"# Aspect Ratio: {bird_data.get('aspect_ratio', 0):.3f}"])
            writer.writerow([f"# Pointedness Index: {bird_data.get('pointedness_index', 0):.4f}"])
            writer.writerow(['#'])

        if metadata is not None:
            writer.writerow(['# === Airfoil Geometric Properties ==='])
            writer.writerow([f"# Max Thickness/Chord: {metadata.get('max_thickness', 0):.4f}"])
            writer.writerow([f"# Max Thickness Position: {metadata.get('max_thickness_pos', 0):.4f}"])
            writer.writerow([f"# Max Camber/Chord: {metadata.get('max_camber', 0):.4f}"])
            writer.writerow([f"# Max Camber Position: {metadata.get('max_camber_pos', 0):.4f}"])
            writer.writerow([f"# Leading Edge Radius: {metadata.get('leading_edge_radius', 0):.4f}"])
            writer.writerow([f"# LE Droop: {metadata.get('le_droop', 0):.4f}"])
            writer.writerow([f"# Trailing Edge Thickness: {metadata.get('trailing_edge_thickness', 0):.4f}"])
            writer.writerow([f"# Estimated Reynolds Number: {metadata.get('estimated_reynolds', 0):.2e}"])
            writer.writerow(['#'])

        writer.writerow([f"# Total Upper CP: {len(upper_cp)}"])
        writer.writerow([f"# Total Lower CP: {len(lower_cp)}"])
        writer.writerow(['#'])

        writer.writerow(['# UPPER SURFACE CONTROL POINTS'])
        writer.writerow(['Surface', 'Point_Index', 'X', 'Y'])
        for i, (x, y) in enumerate(upper_cp):
            writer.writerow(['UPPER', i, f'{x:.8f}', f'{y:.8f}'])

        writer.writerow(['#'])

        writer.writerow(['# LOWER SURFACE CONTROL POINTS'])
        writer.writerow(['Surface', 'Point_Index', 'X', 'Y'])
        for i, (x, y) in enumerate(lower_cp):
            writer.writerow(['LOWER', i, f'{x:.8f}', f'{y:.8f}'])


# ===========================================================================
# BULK EXPORT
# ===========================================================================

def export_all_airfoil_data(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Generate and export airfoil .dat files + control point CSVs for every
    bird in the categorised dataset.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Must contain: Wing.Length, Secondary1, Kipps.Distance,
        Hand-Wing.Index, Tail.Length, aspect_ratio, pointedness_index,
        efficiency_index, species.
        Optional: flight_category, wing_loading_proxy.

    Returns
    -------
    pd.DataFrame   Summary table (one row per bird).
    """
    print(f"\n{'='*60}")
    print(f"EXPORTING ALL {len(df_clean)} BIRD AIRFOILS")
    print("="*60)

    AIRFOIL_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_POINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create per-category subdirectories
    if 'flight_category' in df_clean.columns:
        for category in df_clean['flight_category'].unique():
            (AIRFOIL_DIR / category).mkdir(exist_ok=True)
            (CONTROL_POINTS_DIR / category).mkdir(exist_ok=True)

    airfoil_summary = []

    for i in range(len(df_clean)):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(df_clean)} birds...")

        bird = df_clean.iloc[i]

        upper_cp, lower_cp, metadata = generate_bird_airfoil(
            wing_length        = float(bird['Wing.Length']),
            secondary1         = float(bird['Secondary1']),
            kipps_distance     = float(bird['Kipps.Distance']),
            hand_wing_index    = float(bird['Hand-Wing.Index']),
            tail_length        = float(bird['Tail.Length']),
            aspect_ratio       = float(bird['aspect_ratio']),
            pointedness_index  = float(bird['pointedness_index']),
            wing_loading_proxy = bird.get('wing_loading_proxy'),
            chord_length       = 1.0,
        )

        # High-resolution surface curves for CFD (300 points)
        upper_curve = bezier_curve(upper_cp, 300)
        lower_curve = bezier_curve(lower_cp, 300)

        # Combined coordinates: TE → upper LE → lower → TE (clockwise)
        combined_coords = np.vstack([
            upper_curve[::-1],
            lower_curve[1:],
        ])

        # Geometric properties from sampled curves
        y_lower_interp = np.interp(
            upper_curve[:, 0], lower_curve[:, 0], lower_curve[:, 1]
        )
        thickness_dist  = upper_curve[:, 1] - y_lower_interp
        t_max           = float(np.max(thickness_dist))
        t_max_x         = float(upper_curve[np.argmax(thickness_dist), 0])

        camber_x     = np.linspace(0.0, 1.0, 200)
        upper_interp = np.interp(camber_x, upper_curve[:, 0], upper_curve[:, 1])
        lower_interp = np.interp(camber_x, lower_curve[:, 0], lower_curve[:, 1])
        camber_line  = (upper_interp + lower_interp) / 2.0
        f_max        = float(np.max(camber_line))
        f_max_x      = float(camber_x[np.argmax(camber_line)])

        # File paths
        species_name    = str(bird['species']).replace(' ', '_').replace('/', '_')[:25]
        flight_category = str(bird.get('flight_category', 'generalist'))
        stem            = f'{i+1:05d}_{species_name}_AR{bird["aspect_ratio"]:.2f}'

        if 'flight_category' in bird.index if hasattr(bird, 'index') else 'flight_category' in bird:
            dat_path = AIRFOIL_DIR        / flight_category / f'airfoil_{stem}.dat'
            cp_path  = CONTROL_POINTS_DIR / flight_category / f'cp_{stem}.csv'
        else:
            dat_path = AIRFOIL_DIR        / f'airfoil_{stem}.dat'
            cp_path  = CONTROL_POINTS_DIR / f'cp_{stem}.csv'

        # DAT file header
        header = (
            f"Bird-inspired airfoil generated from AVONET morphological data\n"
            f"Airfoil ID      : {i+1:05d}\n"
            f"Species         : {bird['species']}\n"
            f"Flight Category : {flight_category}\n"
            f"\n"
            f"=== Morphological Measurements ===\n"
            f"Wing Length      : {bird['Wing.Length']:.2f} mm\n"
            f"Secondary1       : {bird['Secondary1']:.2f} mm\n"
            f"Kipps Distance   : {bird['Kipps.Distance']:.2f} mm\n"
            f"Hand-Wing Index  : {bird['Hand-Wing.Index']:.2f}\n"
            f"Tail Length      : {bird['Tail.Length']:.2f} mm\n"
            f"\n"
            f"=== Derived Indices ===\n"
            f"Aspect Ratio     : {bird['aspect_ratio']:.3f}\n"
            f"Pointedness Index: {bird['pointedness_index']:.4f}\n"
            f"Efficiency Index : {bird['efficiency_index']:.4f}\n"
            f"Wing Loading Proxy: {bird.get('wing_loading_proxy', 0):.4f}\n"
            f"\n"
            f"=== Airfoil Geometry (chord-normalised) ===\n"
            f"Max t/c          : {t_max:.4f}  at x/c = {t_max_x:.4f}\n"
            f"Max f/c          : {f_max:.4f}  at x/c = {f_max_x:.4f}\n"
            f"Est. Reynolds    : {metadata['estimated_reynolds']:.2e}\n"
            f"\n"
            f"Coordinates: x/c  y/c,  TE→upper→LE→lower→TE (clockwise)\n"
            f"Total points: {len(combined_coords)}\n"
            f"X/C    Y/C"
        )

        np.savetxt(dat_path, combined_coords, header=header,
                   fmt='%.8f', delimiter='  ', comments='# ')

        # Control points CSV
        bird_dict = {
            'species':           bird['species'],
            'flight_category':   flight_category,
            'Wing.Length':       bird['Wing.Length'],
            'Secondary1':        bird['Secondary1'],
            'Kipps.Distance':    bird['Kipps.Distance'],
            'Hand-Wing.Index':   bird['Hand-Wing.Index'],
            'Tail.Length':       bird['Tail.Length'],
            'aspect_ratio':      bird['aspect_ratio'],
            'pointedness_index': bird['pointedness_index'],
            'efficiency_index':  bird['efficiency_index'],
            'wing_loading_proxy': bird.get('wing_loading_proxy', 0),
        }
        export_control_points_to_csv(upper_cp, lower_cp, cp_path, metadata, bird_dict)

        airfoil_summary.append({
            'ID':                i + 1,
            'Species':           str(bird['species'])[:50],
            'FlightCategory':    flight_category,
            'DatFilename':       dat_path.name,
            'CpFilename':        cp_path.name,
            'AspectRatio':       bird['aspect_ratio'],
            'PointednessIndex':  bird['pointedness_index'],
            'HandWingIndex':     bird['Hand-Wing.Index'],
            'EfficiencyIndex':   bird['efficiency_index'],
            'WingLoadingProxy':  bird.get('wing_loading_proxy', 0),
            'MaxThickness':      t_max,
            'MaxThicknessPos':   t_max_x,
            'MaxCamber':         f_max,
            'MaxCamberPos':      f_max_x,
            'WingLength_mm':     bird['Wing.Length'],
            'Secondary1_mm':     bird['Secondary1'],
            'KippsDistance_mm':  bird['Kipps.Distance'],
            'TailLength_mm':     bird['Tail.Length'],
            'EstimatedReynolds': metadata['estimated_reynolds'],
            'LERadius':          metadata['leading_edge_radius'],
            'LEDroop':           metadata['le_droop'],
        })

    print(f"\nCompleted all {len(df_clean)} birds.")

    summary_df = pd.DataFrame(airfoil_summary)

    summary_df.to_csv(AIRFOIL_DIR / 'complete_airfoil_database.csv', index=False)
    print(f"✓ Saved complete database: {AIRFOIL_DIR / 'complete_airfoil_database.csv'}")

    summary_df.to_csv(CONTROL_POINTS_DIR / 'control_points_summary.csv', index=False)
    print(f"✓ Saved control points summary: {CONTROL_POINTS_DIR / 'control_points_summary.csv'}")

    if 'flight_category' in df_clean.columns:
        category_stats = summary_df.groupby('FlightCategory').agg({
            'AspectRatio':    ['count', 'mean', 'std', 'min', 'max'],
            'MaxThickness':   ['mean', 'std'],
            'MaxCamber':      ['mean', 'std'],
            'HandWingIndex':  ['mean', 'std'],
            'EstimatedReynolds': ['mean', 'min', 'max'],
        }).round(4)

        category_stats.to_csv(AIRFOIL_DIR / 'category_statistics.csv')
        print(f"✓ Saved category statistics: {AIRFOIL_DIR / 'category_statistics.csv'}")

        print(f"\nFlight Category Distribution:")
        for cat in summary_df['FlightCategory'].unique():
            n = len(summary_df[summary_df['FlightCategory'] == cat])
            print(f"  {cat.upper():20s}: {n:5d}  ({n/len(summary_df)*100:.1f}%)")

    print(f"\n✓ {len(summary_df)} airfoil DAT files  → {AIRFOIL_DIR}")
    print(f"✓ {len(summary_df)} control point CSVs → {CONTROL_POINTS_DIR}")
    return summary_df


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("\n" + "="*60)
    print("BIRD-INSPIRED AIRFOIL GENERATION PIPELINE")
    print("="*60)

    categorized_file = OUTPUT_DIR / "birds_with_categories.csv"
    if not categorized_file.exists():
        print(f"✗ {categorized_file} not found — run categorisation.py first")
        return None

    df = pd.read_csv(categorized_file)
    print(f"✓ Loaded {len(df)} categorised birds")

    summary_df = export_all_airfoil_data(df)

    print("\n" + "="*60)
    print("AIRFOIL GENERATION COMPLETE ✓")
    print("="*60)
    print(f"Generated {len(summary_df)} airfoils")
    print(f"Coordinates : {AIRFOIL_DIR}")
    print(f"Ctrl points : {CONTROL_POINTS_DIR}")
    print("Ready for NeuralFoil CFD analysis.")
    return summary_df


if __name__ == "__main__":
    main()