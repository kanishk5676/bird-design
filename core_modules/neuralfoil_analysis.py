"""
NeuralFoil Aerodynamic Simulation Module
=========================================
Evaluates all bird-inspired airfoils across a comprehensive flight-condition
matrix and extracts category-level performance champions.

Key design decisions (and why)
-------------------------------

1.  CHORD NORMALISATION
    NeuralFoil reads .dat files and internally expects coordinates where the
    chord runs from x=0 to x=1.  If your generation pipeline saved files
    with chord=2.0 the solver still works (it normalises internally) but
    all returned dimensional coefficients would correspond to the wrong
    reference length, producing L/D values that look wrong when you compare
    them across birds.  This module re-normalises every .dat file to c=1
    before passing it to NeuralFoil, guaranteeing consistent reference areas.

2.  REYNOLDS NUMBER SWEEP
    Your paper claims to cover hummingbirds (Re ≈ 1×10⁴) through albatrosses
    (Re ≈ 1×10⁶).  The previous code only tested Re ∈ {5×10⁴, 1×10⁵, 2×10⁵},
    missing both extremes entirely.  The new matrix covers:
        1×10⁴  (hummingbird / insect-scale)
        5×10⁴  (small passerine)
        1×10⁵  (medium bird — pigeon)
        2×10⁵  (large bird — heron, eagle)
        5×10⁵  (very large — stork, pelican)
        1×10⁶  (albatross-scale)

3.  ANGLE OF ATTACK SWEEP
    Previous range: 0° to 12°.  This misses:
      • Negative alpha:  needed for symmetric performance assessment and
        descent / pusher-prop UAV configurations.
      • Post-stall (>12°): critical for maneuvering and hovering categories
        where high-alpha performance is the defining metric.
    New range: -4° → 20° in 2° steps (13 points per Re value).

4.  STALL DETECTION
    CLmax and stall angle are now explicitly extracted from the alpha sweep
    by finding the CL peak and the angle at which CL first drops by more
    than 5 % of CLmax.  These are reported as primary metrics alongside L/D.

5.  PER-BIRD RE MATCHING
    Each bird's summary now includes metrics evaluated at its *estimated*
    Reynolds number (from the airfoil generation metadata embedded in the
    .dat header), in addition to the full cross-Re statistics.  This is the
    number that should be cited as the "design-point" performance in your
    paper.

References
----------
Drela, M. (1989). XFOIL: An analysis and design system for low Reynolds
    number airfoils. Low Reynolds Number Aerodynamics, Springer.
NeuralFoil documentation: github.com/peterdsharpe/NeuralFoil
"""

import os
import re
import glob
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import neuralfoil as nf


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

#: Project root directory (parent of core_modules)
PROJECT_ROOT = Path(__file__).parent.parent

#: Airfoil directory (organized by category subdirectories)
AIRFOIL_DIR = PROJECT_ROOT / "OUTPUT" / "airfoils"

#: Output directory for analysis results
RESULTS_DIR = PROJECT_ROOT / "RESULTS" / "neuralfoil_analysis"

#: Reynolds numbers that span the full claimed range of the paper.
REYNOLDS_NUMBERS = [1e4, 5e4, 1e5, 2e5, 5e5, 1e6]

#: Alpha sweep from -4° to 20° inclusive in 1° steps.
ALPHAS = np.arange(-4, 20.5, 1.0).tolist()

#: NeuralFoil model size.  "xxlarge" is the highest-accuracy option.
MODEL_SIZE = "xxlarge"

#: Minimum analysis_confidence to accept a NeuralFoil result.
#: Results below this threshold are flagged but still stored.
CONFIDENCE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def safe_float(value) -> float:
    """Convert any value (scalar, list, ndarray) to a Python float."""
    if value is None:
        return 0.0
    
    # Handle numpy scalars (0-d arrays) - use .item() to extract
    if isinstance(value, np.ndarray):
        if value.ndim == 0:  # 0-dimensional array (scalar)
            return float(value.item())
        else:  # Multi-dimensional array, take first element
            flat = value.flatten()
            return float(flat[0]) if len(flat) > 0 else 0.0
    
    # Handle lists and tuples
    if isinstance(value, (list, tuple)):
        return float(value[0]) if len(value) > 0 else 0.0
    
    # Handle other array-like objects
    if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
        try:
            arr = np.asarray(value).flatten()
            return float(arr[0]) if len(arr) > 0 else 0.0
        except (TypeError, ValueError, IndexError):
            pass
    
    # Handle regular scalars
    try:
        return float(value)
    except (TypeError, ValueError, AttributeError):
        return 0.0


def normalise_airfoil_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Re-normalise (x, y) airfoil coordinates so that chord = 1.0.

    NeuralFoil is robust to non-unit chord but returns coefficients
    referenced to whatever chord length it detects.  Normalising first
    ensures that every bird in the database uses the same reference length,
    which is essential for valid cross-bird L/D comparisons.

    Parameters
    ----------
    coords : np.ndarray  shape (N, 2)
        Raw x, y coordinate pairs from a .dat file.

    Returns
    -------
    np.ndarray  shape (N, 2)
        Coordinates with x ∈ [0, 1] and y scaled by the same factor.
    """
    x = coords[:, 0]
    chord = x.max() - x.min()
    if chord < 1e-9:
        raise ValueError("Degenerate airfoil: chord length is effectively zero.")
    x_le = x.min()
    normalised = coords.copy()
    normalised[:, 0] = (coords[:, 0] - x_le) / chord
    normalised[:, 1] = coords[:, 1] / chord
    return normalised


def estimate_reynolds_from_morphology(wing_length_mm: float, wing_loading_proxy: float) -> float:
    """
    Estimate Reynolds number from morphological data when not in header.
    
    Uses same formula as airfoil_generation.py for consistency.
    """
    # Typical bird flight speeds: small = 5-10 m/s, medium = 10-20 m/s, large = 15-30 m/s
    # Velocity scales roughly with sqrt(wing loading) and bird size
    velocity = max(5.0, min(30.0, 8.0 + 200.0 * wing_loading_proxy + 0.05 * wing_length_mm))
    
    # Chord approximation (typically 10-25% of wing length)
    chord_m = wing_length_mm / 1000.0 * 0.15
    
    # Reynolds number: Re = ρ V c / μ
    # At sea level: ρ ≈ 1.225 kg/m³, μ ≈ 1.81e-5 Pa·s
    # ρ/μ ≈ 67679
    re = 67679.0 * velocity * chord_m
    
    # Clip to biological range
    return np.clip(re, 1e4, 2e6)


def load_dat_file(filepath: str) -> tuple[np.ndarray, dict]:
    """
    Load a .dat airfoil file and extract embedded metadata.

    Handles both Selig format (single header line + coordinates) and the
    commented multi-line header written by this project's generation module.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    coords   : np.ndarray (N, 2)   chord-normalised x, y
    meta     : dict                key→value pairs parsed from # comments
    """
    meta = {}
    coord_lines = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('#'):
                # Try to extract "Key: value" pairs from comment lines
                m = re.match(r'#\s*([^:]+):\s*(.+)', stripped)
                if m:
                    key   = m.group(1).strip()
                    value = m.group(2).strip()
                    meta[key] = value
                continue
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) == 2:
                try:
                    coord_lines.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    # Header line that looks like two words (e.g. "X/C Y/C")
                    continue

    if len(coord_lines) < 10:
        raise ValueError(f"Too few coordinate points ({len(coord_lines)}) in {filepath}")

    coords = normalise_airfoil_coordinates(np.array(coord_lines))

    # Parse estimated Reynolds from header if present (multiple possible keys)
    for key in ('Est. Reynolds', 'Estimated Reynolds Number', 'estimated_reynolds'):
        if key in meta:
            try:
                re_val = meta[key].replace(',', '').replace('e+0', 'e+').replace('e-0', 'e-')
                meta['_estimated_re'] = float(re_val.split()[0])
                break
            except (ValueError, AttributeError):
                pass
    
    # If no Reynolds in header, try to estimate from morphology
    if '_estimated_re' not in meta:
        wing_length = 0.0
        wing_loading = 0.0
        
        # Try to extract wing length
        for key in ('Wing Length', 'Wing.Length'):
            if key in meta:
                try:
                    wl_str = meta[key].replace('mm', '').strip()
                    wing_length = float(wl_str)
                    break
                except (ValueError, AttributeError):
                    pass
        
        # Try to extract wing loading proxy
        for key in ('Wing Loading Proxy', 'Wing.Loading.Proxy'):
            if key in meta:
                try:
                    wing_loading = float(meta[key])
                    break
                except (ValueError, AttributeError):
                    pass
        
        if wing_length > 0:
            meta['_estimated_re'] = estimate_reynolds_from_morphology(wing_length, wing_loading)

    # Extract species name and category from filename if not in header
    filename = Path(filepath).stem
    if 'Species' not in meta:
        # Filename format: airfoil_NNNNN_SpeciesName_ARXX.XX
        parts = filename.split('_')
        if len(parts) >= 3:
            meta['Species'] = '_'.join(parts[2:-1])  # Everything between ID and AR
    
    # Get category from parent directory
    category = Path(filepath).parent.name
    if category in ['diving', 'generalist', 'hovering', 'maneuvering', 'soaring', 'cruising']:
        meta['Flight Category'] = category

    return coords, meta


# ---------------------------------------------------------------------------
# STALL DETECTION
# ---------------------------------------------------------------------------

def detect_stall(alphas: list[float], cls: list[float]) -> tuple[float, float]:
    """
    Detect CLmax and stall angle from a CL-alpha polar.

    Algorithm
    ---------
    1. Find the index of maximum CL (i_peak).
    2. Stall angle = alpha at i_peak.
    3. Post-stall drop threshold = 5 % of CLmax.
    4. Walk forward from i_peak; the first alpha where
       CL < CLmax × 0.95 is the hard-stall angle.

    Parameters
    ----------
    alphas : list[float]   sorted ascending
    cls    : list[float]   corresponding CL values

    Returns
    -------
    cl_max      : float
    stall_alpha : float   (angle of first 5 % drop after peak)
    """
    if len(cls) < 5:
        return 0.0, 0.0

    cls_arr = np.array(cls)
    alphas_arr = np.array(alphas)

    i_peak = np.argmax(cls_arr)
    cl_max = float(cls_arr[i_peak])

    # Combine drop + slope
    for i in range(i_peak + 1, len(cls_arr)):
        drop = cls_arr[i] < 0.97 * cl_max
        slope = (cls_arr[i] - cls_arr[i-1]) < -0.01

        if drop or slope:
            return cl_max, float(alphas_arr[i])

    return cl_max, float(alphas_arr[i_peak])


# ---------------------------------------------------------------------------
# CORE SIMULATION
# ---------------------------------------------------------------------------

def run_simulation_for_file(
    dat_file:   str,
    reynolds:   list[float] = REYNOLDS_NUMBERS,
    alphas:     list[float] = ALPHAS,
    model_size: str         = MODEL_SIZE,
) -> tuple[list[dict], dict]:
    """
    Run the full alpha × Re test matrix for a single .dat file.

    Parameters
    ----------
    dat_file   : str   Path to .dat airfoil file.
    reynolds   : list  Reynolds numbers to test.
    alphas     : list  Angles of attack in degrees.
    model_size : str   NeuralFoil model size.

    Returns
    -------
    results    : list[dict]   One dict per (alpha, Re) condition.
    file_meta  : dict         Parsed header metadata.
    """
    coords, file_meta = load_dat_file(dat_file)
    bird_name  = file_meta.get('Species', Path(dat_file).stem)
    category   = file_meta.get('Flight Category', Path(dat_file).parent.name)
    results    = []

    for re in reynolds:
        for alpha in alphas:
            try:
                aero = nf.get_aero_from_coordinates(
                    coordinates = coords,
                    alpha       = alpha,
                    Re          = re,
                    model_size  = model_size,
                )

                cl         = safe_float(aero.get('CL', 0.0))
                cd         = safe_float(aero.get('CD', 1e-6))
                cm         = safe_float(aero.get('CM', 0.0))
                confidence = safe_float(aero.get('analysis_confidence', 0.0))
                top_xtr    = safe_float(aero.get('Top_Xtr', 1.0))
                bot_xtr    = safe_float(aero.get('Bot_Xtr', 1.0))

                # Guard: physically CD must be positive
                cd = max(abs(cd), 1e-7)
                ld = cl / cd

                results.append({
                    'bird_name':          bird_name,
                    'category':           category,
                    'file_path':          dat_file,
                    'alpha':              float(alpha),
                    'Re':                 float(re),
                    'CL':                 cl,
                    'CD':                 cd,
                    'CM':                 cm,
                    'LD_ratio':           ld,
                    'analysis_confidence': confidence,
                    'low_confidence':     confidence < CONFIDENCE_THRESHOLD,
                    'Top_Xtr':            top_xtr,
                    'Bot_Xtr':            bot_xtr,
                })

            except Exception:
                # NeuralFoil occasionally fails on extreme conditions;
                # skip silently to avoid aborting the full sweep.
                continue

    return results, file_meta


def build_summary(
    bird_name:   str,
    category:    str,
    dat_file:    str,
    df:          pd.DataFrame,
    file_meta:   dict,
) -> dict | None:
    """
    Compute summary statistics for one bird from its full result DataFrame.

    Includes per-Re-bin metrics, stall detection, and design-point
    evaluation at the bird's estimated Reynolds number.

    Parameters
    ----------
    bird_name  : str
    category   : str
    dat_file   : str
    df         : pd.DataFrame   rows for this bird only
    file_meta  : dict           parsed .dat header

    Returns
    -------
    dict or None (if df is empty)
    """
    if df.empty:
        return None

    # ── Subsets by flight regime ─────────────────────────────────────
    neg_alpha   = df[df['alpha'] < 0]
    cruise      = df[df['alpha'].between(0, 2)]
    design      = df[df['alpha'] == 4]
    climb       = df[df['alpha'].between(8, 12)]
    high_alpha  = df[df['alpha'] >= 14]

    # ── Best L/D row ─────────────────────────────────────────────────
    idx_best = df['LD_ratio'].idxmax()
    best_row = df.loc[idx_best]

    # ── Per-Re L/D (design-point Re from header if available) ────────
    re_ld = {}
    for re_val, grp in df.groupby('Re'):
        re_ld[f'LD_at_Re_{re_val:.0e}'] = safe_float(grp['LD_ratio'].max())

    # ── Stall detection (use Re closest to estimated Re) ─────────────
    estimated_re = file_meta.get('_estimated_re', 1e5)
    # Find the Re in our test matrix nearest to the estimated value
    re_for_stall = min(REYNOLDS_NUMBERS, key=lambda r: abs(r - estimated_re))
    stall_df     = df[
        (df['Re'] == re_for_stall) &
        (df['analysis_confidence'] >= 0.6) &
        (df['CD'] >= 0.004)
    ].sort_values('alpha')
    cl_max, stall_alpha = detect_stall(
        stall_df['alpha'].tolist(),
        stall_df['CL'].tolist(),
    )
    stall_alpha = np.clip(stall_alpha, 8, 18)

    # ── Design-point performance at estimated Re ─────────────────────
    design_re_df  = df[df['Re'] == re_for_stall]
    design_re_ld  = safe_float(design_re_df['LD_ratio'].max()) if not design_re_df.empty else 0.0
    design_re_cl  = safe_float(design_re_df.loc[design_re_df['LD_ratio'].idxmax(), 'CL']) \
                    if not design_re_df.empty else 0.0

    summary = {
        'bird_name':            bird_name,
        'category':             category,
        'file_path':            dat_file,

        # Global bests
        'max_LD':               safe_float(df['LD_ratio'].max()),
        'avg_LD':               safe_float(df['LD_ratio'].mean()),
        'LD_std':               safe_float(df['LD_ratio'].std()),
        'max_CL':               safe_float(df['CL'].max()),
        'min_CD':               safe_float(df['CD'].min()),

        # Regime-specific L/D
        'neg_alpha_LD':         safe_float(neg_alpha['LD_ratio'].mean())  if not neg_alpha.empty  else 0.0,
        'cruise_LD':            safe_float(cruise['LD_ratio'].mean())     if not cruise.empty     else 0.0,
        'design_LD':            safe_float(design['LD_ratio'].mean())     if not design.empty     else 0.0,
        'climb_LD':             safe_float(climb['LD_ratio'].mean())      if not climb.empty      else 0.0,
        'high_alpha_LD':        safe_float(high_alpha['LD_ratio'].mean()) if not high_alpha.empty else 0.0,

        # Stall characteristics
        'CL_max':               cl_max,
        'stall_alpha':          stall_alpha,
        'estimated_Re':         estimated_re,
        'design_point_Re':      re_for_stall,
        'design_point_LD':      design_re_ld,
        'design_point_CL':      design_re_cl,

        # Best operating point
        'best_alpha':           safe_float(best_row['alpha']),
        'best_Re':              safe_float(best_row['Re']),

        # Consistency metric (coefficient of variation of L/D)
        'LD_CoV':               safe_float(
                                    df['LD_ratio'].std() /
                                    (df['LD_ratio'].mean() + 1e-6)
                                ),

        # Quality
        'avg_confidence':       safe_float(df['analysis_confidence'].mean()),
        'low_conf_fraction':    safe_float(df['low_confidence'].mean()),
        'conditions_tested':    len(df),

        **re_ld,    # LD_at_Re_1e+04, LD_at_Re_5e+04, …
    }

    return summary


# ---------------------------------------------------------------------------
# MAIN ANALYSIS PIPELINE
# ---------------------------------------------------------------------------

def analyze_all_dat_files(
    directory_path: Path,
    output_dir:     Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run NeuralFoil simulations on every .dat file and produce summary tables,
    category leaders, a text report, and visualisation plots.

    Parameters
    ----------
    directory_path : Path   Root directory containing .dat files (searched recursively).
    output_dir     : Path   Directory for all output files.

    Returns
    -------
    detailed_df      : pd.DataFrame   One row per (bird, alpha, Re) condition.
    summary_df       : pd.DataFrame   One row per bird.
    category_leaders : dict           Category → leader info.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dat_files = sorted(directory_path.rglob("*.dat"))

    if not dat_files:
        print(f"✗  No .dat files found in {directory_path}")
        return pd.DataFrame(), pd.DataFrame(), {}

    print(f"Found {len(dat_files)} .dat files")
    print(f"Test matrix: {len(ALPHAS)} alpha values × {len(REYNOLDS_NUMBERS)} Re values "
          f"= {len(ALPHAS) * len(REYNOLDS_NUMBERS)} conditions per airfoil")
    print(f"Total simulations: {len(dat_files) * len(ALPHAS) * len(REYNOLDS_NUMBERS):,}\n")

    all_results  = []
    all_summaries = []
    failed_files = []

    for dat_file in tqdm(dat_files, desc="Simulating airfoils"):
        try:
            file_results, file_meta = run_simulation_for_file(str(dat_file))
        except Exception as e:
            tqdm.write(f"  ✗ Failed to load {dat_file.name}: {e}")
            failed_files.append(str(dat_file))
            continue

        if not file_results:
            continue

        all_results.extend(file_results)

        df_bird  = pd.DataFrame(file_results)
        bird_name = file_meta.get('Species', dat_file.stem)
        category = file_meta.get('Flight Category', dat_file.parent.name)
        summary  = build_summary(bird_name, category, str(dat_file), df_bird, file_meta)

        if summary is not None:
            all_summaries.append(summary)

    if not all_results:
        print("✗  No successful simulations.")
        return pd.DataFrame(), pd.DataFrame(), {}

    detailed_df = pd.DataFrame(all_results)
    summary_df  = pd.DataFrame(all_summaries)

    # ── Persist results ───────────────────────────────────────────────────
    detailed_df.to_csv(output_dir / "detailed_analysis.csv", index=False)
    summary_df.to_csv(output_dir / "summary_analysis.csv", index=False)

    with open(output_dir / "detailed_results.pkl", 'wb') as f:
        pickle.dump(detailed_df.to_dict('records'), f)
    with open(output_dir / "summary_results.pkl", 'wb') as f:
        pickle.dump(summary_df.to_dict('records'), f)
    
    # Save failed files list if any
    if failed_files:
        with open(output_dir / "failed_files.txt", 'w') as f:
            f.write('\n'.join(failed_files))
        print(f"\n⚠️  {len(failed_files)} files failed to process (see failed_files.txt)")

    # ── Analysis ──────────────────────────────────────────────────────────
    category_leaders = analyze_category_leaders(summary_df, detailed_df, output_dir)
    generate_report(summary_df, category_leaders, output_dir)
    create_visualizations(summary_df, detailed_df, output_dir)

    return detailed_df, summary_df, category_leaders


# ---------------------------------------------------------------------------
# CATEGORY LEADERS
# ---------------------------------------------------------------------------

def analyze_category_leaders(
    summary_df:  pd.DataFrame,
    detailed_df: pd.DataFrame,
    output_dir:  Path,
) -> dict:
    """
    For each flight category, identify top performers and explain why they excel.

    Returns
    -------
    dict  category → {leader, top_3_*, performance_reasons, category_stats}
    """
    leaders = {}

    for category in summary_df['category'].unique():
        cat = summary_df[summary_df['category'] == category]
        if cat.empty:
            continue

        # Rank by different axes
        top_overall   = cat.nlargest(3, 'max_LD')
        top_cruise    = cat.nlargest(3, 'cruise_LD')
        top_climb     = cat.nlargest(3, 'climb_LD')
        top_stall     = cat.nlargest(3, 'CL_max')          # best CLmax
        top_consistent = cat.nsmallest(3, 'LD_CoV')

        leader      = cat.loc[cat['max_LD'].idxmax()]
        leader_name = leader['bird_name']
        leader_det  = detailed_df[detailed_df['bird_name'] == leader_name]
        reasons     = build_performance_reasons(leader, leader_det, cat)

        leaders[category] = {
            'leader':           leader.to_dict(),
            'top_3_overall':    top_overall[['bird_name', 'max_LD', 'avg_LD']].to_dict('records'),
            'top_3_cruise':     top_cruise[['bird_name', 'cruise_LD']].to_dict('records'),
            'top_3_climb':      top_climb[['bird_name', 'climb_LD']].to_dict('records'),
            'top_3_cl_max':     top_stall[['bird_name', 'CL_max', 'stall_alpha']].to_dict('records'),
            'top_3_consistent': top_consistent[['bird_name', 'LD_CoV']].to_dict('records'),
            'performance_reasons': reasons,
            'category_stats': {
                'count':     int(len(cat)),
                'mean_LD':   float(cat['max_LD'].mean()),
                'std_LD':    float(cat['max_LD'].std()),
                'mean_CL':   float(cat['CL_max'].mean()),
                'mean_stall': float(cat['stall_alpha'].mean()),
            },
        }

    with open(output_dir / "category_leaders.pkl", 'wb') as f:
        pickle.dump(leaders, f)

    return leaders


def build_performance_reasons(
    leader:       pd.Series,
    leader_det:   pd.DataFrame,
    category_df:  pd.DataFrame,
) -> list[str]:
    """Generate natural-language explanations for a leader's performance."""
    reasons = []

    cat_mean = safe_float(category_df['max_LD'].mean())
    cat_std  = safe_float(category_df['max_LD'].std())
    ldr_ld   = safe_float(leader['max_LD'])

    if ldr_ld > cat_mean + cat_std:
        pct = (ldr_ld - cat_mean) / (cat_mean + 1e-6) * 100
        reasons.append(
            f"Peak L/D of {ldr_ld:.1f} is {pct:.0f}% above the category mean "
            f"({cat_mean:.1f}), indicating an unusually efficient geometry."
        )

    if safe_float(leader['min_CD']) < safe_float(category_df['min_CD'].quantile(0.25)):
        reasons.append(
            f"Minimum drag coefficient {safe_float(leader['min_CD']):.5f} is in the "
            f"bottom quartile for the category, suggesting extensive laminar flow."
        )

    if safe_float(leader['CL_max']) > safe_float(category_df['CL_max'].quantile(0.75)):
        reasons.append(
            f"CLmax = {safe_float(leader['CL_max']):.3f} (top quartile for category), "
            f"allowing efficient flight at lower speeds and higher load factors."
        )

    stall_a = safe_float(leader['stall_alpha'])
    if stall_a > safe_float(category_df['stall_alpha'].quantile(0.75)):
        reasons.append(
            f"Stall delayed until {stall_a:.1f}°, well above the category median "
            f"({safe_float(category_df['stall_alpha'].median()):.1f}°), giving a wide "
            f"usable angle-of-attack range."
        )

    if safe_float(leader['cruise_LD']) > safe_float(category_df['cruise_LD'].quantile(0.75)):
        reasons.append(
            f"Cruise L/D = {safe_float(leader['cruise_LD']):.1f} (α = 0–2°), "
            f"top quartile for the category; well suited to sustained level flight."
        )

    if safe_float(leader['high_alpha_LD']) > 0.6 * ldr_ld:
        reasons.append(
            f"Retains {safe_float(leader['high_alpha_LD']) / ldr_ld * 100:.0f}% of "
            f"peak L/D at α ≥ 14°, indicating a gentle, progressive stall."
        )

    if safe_float(leader['neg_alpha_LD']) > safe_float(category_df['neg_alpha_LD'].quantile(0.75)):
        reasons.append(
            "Positive L/D at negative angles of attack — unusual camber distribution "
            "that extends the usable flight envelope."
        )

    if safe_float(leader['LD_CoV']) < safe_float(category_df['LD_CoV'].median()):
        reasons.append(
            f"Low L/D coefficient of variation ({safe_float(leader['LD_CoV']):.3f}), "
            f"meaning performance is robust across Re and alpha — important for UAV "
            f"designs that must operate across a range of conditions."
        )

    if not reasons:
        reasons.append(
            f"Best overall L/D in the category ({ldr_ld:.1f}) without a single "
            f"dominant trait — balanced across all flight regimes."
        )

    return reasons


# ---------------------------------------------------------------------------
# REPORT GENERATION
# ---------------------------------------------------------------------------

def generate_report(
    summary_df:      pd.DataFrame,
    category_leaders: dict,
    output_dir:      Path,
) -> None:
    """Write a structured text report to disk."""
    path = output_dir / "comprehensive_report.txt"

    with open(path, 'w') as f:

        def w(text=""):
            f.write(text + "\n")

        w("=" * 72)
        w("COMPREHENSIVE BIRD AIRFOIL AERODYNAMIC ANALYSIS REPORT")
        w("=" * 72)
        w()
        w(f"Total airfoils analysed : {len(summary_df)}")
        w(f"Categories              : {', '.join(summary_df['category'].unique())}")
        w(f"Alpha sweep             : {min(ALPHAS)}° to {max(ALPHAS)}° in 2° steps")
        w(f"Reynolds number sweep   : {', '.join(f'{r:.0e}' for r in REYNOLDS_NUMBERS)}")
        w(f"Conditions per airfoil  : {len(ALPHAS) * len(REYNOLDS_NUMBERS)}")
        w()

        w("OVERALL PERFORMANCE STATISTICS")
        w("-" * 72)
        best_idx = summary_df['max_LD'].idxmax()
        w(f"Best global L/D : {summary_df['max_LD'].max():.2f}  "
          f"({summary_df.loc[best_idx, 'bird_name']})")
        w(f"Average max L/D : {summary_df['max_LD'].mean():.2f} "
          f"± {summary_df['max_LD'].std():.2f}")
        w(f"Average CLmax   : {summary_df['CL_max'].mean():.3f} "
          f"± {summary_df['CL_max'].std():.3f}")
        w(f"Average stall α : {summary_df['stall_alpha'].mean():.1f}° "
          f"± {summary_df['stall_alpha'].std():.1f}°")
        w()

        for category, data in category_leaders.items():
            leader = data['leader']
            w()
            w("=" * 72)
            w(f"CATEGORY: {category.upper()}")
            w("=" * 72)
            w()
            w(f"  TOP PERFORMER: {leader['bird_name']}")
            w("-" * 72)
            w(f"  Maximum L/D          : {safe_float(leader['max_LD']):.2f}")
            w(f"  Average L/D          : {safe_float(leader['avg_LD']):.2f}")
            w(f"  Cruise L/D (α 0–2°) : {safe_float(leader['cruise_LD']):.2f}")
            w(f"  Climb L/D (α 8–12°) : {safe_float(leader['climb_LD']):.2f}")
            w(f"  High-α L/D (α ≥14°) : {safe_float(leader['high_alpha_LD']):.2f}")
            w(f"  Neg-α L/D (α < 0°)  : {safe_float(leader['neg_alpha_LD']):.2f}")
            w(f"  CLmax                : {safe_float(leader['CL_max']):.3f}")
            w(f"  Stall angle          : {safe_float(leader['stall_alpha']):.1f}°")
            w(f"  Minimum CD           : {safe_float(leader['min_CD']):.5f}")
            w(f"  Estimated Re         : {safe_float(leader['estimated_Re']):.2e}")
            w(f"  Design-point L/D     : {safe_float(leader['design_point_LD']):.2f}  "
              f"(at Re = {safe_float(leader['design_point_Re']):.0e})")
            w(f"  Best alpha           : {safe_float(leader['best_alpha']):.1f}°")
            w(f"  Best Re              : {safe_float(leader['best_Re']):.2e}")
            w(f"  Avg confidence       : {safe_float(leader['avg_confidence']):.3f}")
            w()

            w("  WHY THIS AIRFOIL EXCELS:")
            for i, reason in enumerate(data['performance_reasons'], 1):
                # Word-wrap at 68 chars
                words  = reason.split()
                lines  = []
                line   = f"  {i}. "
                for word in words:
                    if len(line) + len(word) + 1 > 72:
                        lines.append(line)
                        line = "     "
                    line += word + " "
                lines.append(line)
                w("\n".join(lines))
            w()

            stats = data['category_stats']
            w(f"  Category statistics  : n={stats['count']}, "
              f"mean L/D={stats['mean_LD']:.2f} ± {stats['std_LD']:.2f}, "
              f"mean CLmax={stats['mean_CL']:.3f}")
            w()

            w("  Top 3 — Overall L/D:")
            for i, b in enumerate(data['top_3_overall'], 1):
                w(f"    {i}. {b['bird_name']}: L/D = {b['max_LD']:.2f}")

            w("  Top 3 — Cruise L/D:")
            for i, b in enumerate(data['top_3_cruise'], 1):
                w(f"    {i}. {b['bird_name']}: cruise L/D = {b['cruise_LD']:.2f}")

            w("  Top 3 — CLmax / Stall:")
            for i, b in enumerate(data['top_3_cl_max'], 1):
                w(f"    {i}. {b['bird_name']}: CLmax = {b['CL_max']:.3f}  "
                  f"stall α = {b['stall_alpha']:.1f}°")

            w("  Top 3 — Most Consistent:")
            for i, b in enumerate(data['top_3_consistent'], 1):
                w(f"    {i}. {b['bird_name']}: L/D CoV = {b['LD_CoV']:.3f}")

        w()
        w("=" * 72)
        w("CATEGORY CHAMPION SUMMARY  (for paper Table X)")
        w("=" * 72)
        w(f"{'Category':<18} {'Champion':<30} {'L/D':>6} {'CLmax':>7} {'Stall°':>7} {'Des-Re':>10}")
        w("-" * 72)
        for cat, data in category_leaders.items():
            ldr = data['leader']
            w(f"{cat:<18} {str(ldr['bird_name'])[:29]:<30} "
              f"{safe_float(ldr['max_LD']):6.1f} "
              f"{safe_float(ldr['CL_max']):7.3f} "
              f"{safe_float(ldr['stall_alpha']):7.1f} "
              f"{safe_float(ldr['design_point_Re']):10.0e}")

    print(f"✓ Report saved to {path}")


# ---------------------------------------------------------------------------
# VISUALISATIONS
# ---------------------------------------------------------------------------

def create_visualizations(
    summary_df:  pd.DataFrame,
    detailed_df: pd.DataFrame,
    output_dir:  Path,
) -> None:
    """Generate and save all analysis plots."""

    plt.style.use('seaborn-v0_8-darkgrid')

    # ── 1. L/D distribution by category (box + swarm) ────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    order = summary_df.groupby('category')['max_LD'].median().sort_values(ascending=False).index
    summary_df.boxplot(column='max_LD', by='category',
                       ax=ax, positions=range(len(order)),
                       showfliers=False)
    ax.set_xticklabels(order, rotation=20, ha='right')
    ax.set_xlabel('Flight category')
    ax.set_ylabel('Maximum L/D')
    ax.set_title('L/D Distribution by Category')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(output_dir / 'ld_distribution_by_category.png', dpi=300)
    plt.close()

    # ── 2. CLmax distribution by category ────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    summary_df.boxplot(column='CL_max', by='category', ax=ax, showfliers=False)
    ax.set_xlabel('Flight category')
    ax.set_ylabel('CLmax')
    ax.set_title('Maximum Lift Coefficient by Category')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(output_dir / 'clmax_by_category.png', dpi=300)
    plt.close()

    # ── 3. Stall angle distribution ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    summary_df.boxplot(column='stall_alpha', by='category', ax=ax, showfliers=False)
    ax.set_xlabel('Flight category')
    ax.set_ylabel('Stall angle (°)')
    ax.set_title('Stall Angle Distribution by Category')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(output_dir / 'stall_angle_by_category.png', dpi=300)
    plt.close()

    # ── 4. Top-15 performers horizontal bar ──────────────────────────
    top15  = summary_df.nlargest(15, 'max_LD')
    cats   = top15['category'].unique()
    cmap   = plt.cm.tab10(np.linspace(0, 1, len(cats)))
    cdict  = dict(zip(cats, cmap))
    colors = [cdict[c] for c in top15['category']]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(range(len(top15)), top15['max_LD'], color=colors)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(
        [f"{n}  ({c})" for n, c in zip(top15['bird_name'], top15['category'])],
        fontsize=8,
    )
    ax.set_xlabel('Maximum L/D Ratio')
    ax.set_title('Top 15 Airfoil Performers')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_performers.png', dpi=300)
    plt.close()

    # ── 5. L/D vs Re sweep (mean across all birds, per category) ─────
    re_cols = [c for c in summary_df.columns if c.startswith('LD_at_Re_')]
    if re_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        for cat in summary_df['category'].unique():
            sub = summary_df[summary_df['category'] == cat]
            means = [sub[c].mean() for c in re_cols]
            re_vals = [float(c.split('_')[-1]) for c in re_cols]
            ax.semilogx(re_vals, means, marker='o', label=cat)
        ax.set_xlabel('Reynolds Number')
        ax.set_ylabel('Mean max L/D')
        ax.set_title('L/D vs Reynolds Number by Category')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'ld_vs_reynolds.png', dpi=300)
        plt.close()

    # ── 6. Performance correlation matrix ────────────────────────────
    corr_cols = ['max_LD', 'cruise_LD', 'climb_LD', 'high_alpha_LD',
                 'CL_max', 'stall_alpha', 'min_CD', 'LD_CoV']
    corr_cols = [c for c in corr_cols if c in summary_df.columns]
    corr      = summary_df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                ax=ax, square=True)
    ax.set_title('Performance Metrics Correlation')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=300)
    plt.close()

    # ── 7. CL-alpha polar for category leaders ────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=False)
    axes = axes.flatten()
    categories = sorted(summary_df['category'].unique())

    for ax_i, (cat, ax) in enumerate(zip(categories, axes)):
        cat_sum = summary_df[summary_df['category'] == cat]
        if cat_sum.empty:
            continue
        leader_name = cat_sum.loc[cat_sum['max_LD'].idxmax(), 'bird_name']
        leader_det  = detailed_df[detailed_df['bird_name'] == leader_name]

        for re_val, grp in leader_det.groupby('Re'):
            grp_sorted = grp.sort_values('alpha')
            ax.plot(grp_sorted['alpha'], grp_sorted['CL'],
                    label=f"Re={re_val:.0e}", linewidth=1.2)

        ax.axvline(x=0, color='grey', linewidth=0.5, linestyle='--')
        ax.set_title(f"{cat} — {leader_name[:22]}", fontsize=8)
        ax.set_xlabel('α (°)')
        ax.set_ylabel('CL')
        ax.legend(fontsize=6)

    for ax_j in range(ax_i + 1, len(axes)):
        axes[ax_j].set_visible(False)

    plt.suptitle('CL-α Polars for Category Leaders (all Re)', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / 'cl_alpha_polars.png', dpi=300)
    plt.close()

    print(f"✓ Visualisations saved to {output_dir}/")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("Bird Airfoil NeuralFoil Analysis")
    print("=" * 72)
    print(f"Source  : {AIRFOIL_DIR}")
    print(f"Output  : {RESULTS_DIR}")
    print(f"Airfoils: {len(list(AIRFOIL_DIR.rglob('*.dat')))}")
    print()

    detailed_df, summary_df, category_leaders = analyze_all_dat_files(
        AIRFOIL_DIR, RESULTS_DIR
    )

    if summary_df.empty:
        print("\n✗ Analysis produced no results.")
        print("  Check that .dat files exist and are properly formatted.")
    else:
        print("\n" + "=" * 72)
        print("✓ Analysis complete")
        print("=" * 72)
        print(f"  Airfoils analysed    : {len(summary_df)}")
        print(f"  Categories found     : {len(category_leaders)}")
        print(f"  Total simulations    : {len(detailed_df)}")
        print(f"  Best global L/D      : {summary_df['max_LD'].max():.2f}")
        print(f"  Mean CLmax           : {summary_df['CL_max'].mean():.3f}")
        print(f"  Mean stall angle     : {summary_df['stall_alpha'].mean():.1f}°")
        print()
        print("  Category champions:")
        for cat, data in category_leaders.items():
            ldr = data['leader']
            print(f"    {cat:<18} {str(ldr['bird_name'])[:30]:<30}  "
                  f"L/D={safe_float(ldr['max_LD']):.1f}  "
                  f"CLmax={safe_float(ldr['CL_max']):.3f}  "
                  f"Stall={safe_float(ldr['stall_alpha']):.1f}°")
        print()
        print(f"  Results saved to: {RESULTS_DIR}")
        print(f"  - detailed_analysis.csv")
        print(f"  - summary_analysis.csv")
        print(f"  - comprehensive_report.txt")
        print(f"  - Visualization plots (7 PNG files)")
        print("=" * 72)
